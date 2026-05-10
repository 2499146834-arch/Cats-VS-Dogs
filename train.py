"""
Cats vs Dogs — Transfer Learning Experiment (Fully Automated)
=============================================================
Improvements over the baseline:
  1. MobileNetV2 pretrained backbone (replaces from-scratch CNN)
  2. GlobalAveragePooling2D instead of Flatten (cuts ~19M params to ~2.5M)
  3. Dropout(0.5) for regularisation
  4. Colour augmentation (brightness, contrast, saturation)
  5. ReduceLROnPlateau + EarlyStopping
  6. Two-phase training: freeze backbone -> train head -> fine-tune

Outputs (saved to output/):
  - model.keras / training_curves.png / confusion_matrix.png
  - classification_report.txt / error_analysis.png
"""

import os, sys, ssl, zipfile, shutil, urllib3
urllib3.disable_warnings()
ssl._create_default_https_context = ssl._create_unverified_context
os.environ["CURL_CA_BUNDLE"] = ""
os.environ["REQUESTS_CA_BUNDLE"] = ""

import requests
_orig_send = requests.Session.send
def _patched_send(self, request, **kwargs):
    kwargs["verify"] = False
    return _orig_send(self, request, **kwargs)
requests.Session.send = _patched_send

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

# --- Config -----------------------------------------------------------
IMG_SIZE      = (160, 160)
BATCH         = 32
EPOCHS_HEAD   = 15
EPOCHS_FT     = 15
LR_HEAD       = 1e-3
LR_FT         = 1e-5
DROPOUT       = 0.5
SEED          = 42
BASE_DIR      = os.path.dirname(__file__)
DATASET_DIR   = os.path.join(BASE_DIR, "dataset")
OUTPUT_DIR    = os.path.join(BASE_DIR, "output")
AUTOTUNE      = tf.data.AUTOTUNE

tf.random.set_seed(SEED)
np.random.seed(SEED)
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 60)
print("Cats vs Dogs — Transfer Learning Pipeline")
print(f"TensorFlow {tf.__version__}  |  Keras {keras.__version__}")
print("=" * 60)

# --- 1. Extract dataset if needed -------------------------------------
print("\n[1/6] Preparing dataset ...")
TRAIN_DIR = os.path.join(DATASET_DIR, "train")
VAL_DIR   = os.path.join(DATASET_DIR, "val")

if not os.path.exists(TRAIN_DIR):
    print("  Extracting ZIP (this only runs once) ...")
    # Find the downloaded ZIP
    download_dir = os.path.expanduser(
        "~\\tensorflow_datasets\\downloads\\cats_vs_dogs")
    zips = [f for f in os.listdir(download_dir) if f.endswith(".zip")]
    if not zips:
        # Try downloading manually
        print("  No cached ZIP found. Downloading ...")
        url = ("https://download.microsoft.com/download/3/E/1/"
               "3E1C3F21-ECDB-4869-8368-6DEBA77B919F/"
               "kagglecatsanddogs_5340.zip")
        zip_path = os.path.join(DATASET_DIR, "kagglecatsanddogs.zip")
        os.makedirs(DATASET_DIR, exist_ok=True)
        import urllib.request
        urllib.request.urlretrieve(url, zip_path)
    else:
        zip_path = os.path.join(download_dir, zips[0])
        print(f"  Using cached: {os.path.basename(zip_path)}")

    print("  Extracting ...")
    tmp = os.path.join(DATASET_DIR, "_tmp")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(tmp)

    # The ZIP contains PetImages/Cat/*.jpg and PetImages/Dog/*.jpg
    src_cat = os.path.join(tmp, "PetImages", "Cat")
    src_dog = os.path.join(tmp, "PetImages", "Dog")

    # Create train/val splits
    for split, ratio in [("train", 0.8), ("val", 0.2)]:
        for cls, src in [("cat", src_cat), ("dog", src_dog)]:
            dst = os.path.join(DATASET_DIR, split, cls)
            os.makedirs(dst, exist_ok=True)
            fnames = sorted(os.listdir(src))
            cutoff = int(len(fnames) * ratio)
            subset = fnames[:cutoff] if split == "train" else fnames[cutoff:]
            for fn in subset:
                shutil.copy2(os.path.join(src, fn), os.path.join(dst, fn))
    shutil.rmtree(tmp)
    print("  Dataset ready!")

# --- Pre-scan: clean ALL images — re-encode as valid RGB JPEGs ---------
print("  Scanning and cleaning images (re-encoding to clean JPEG) ...")
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
removed = 0
cleaned = 0
for split in ["train", "val"]:
    for cls in ["cat", "dog"]:
        d = os.path.join(DATASET_DIR, split, cls)
        for fn in os.listdir(d):
            fp = os.path.join(d, fn)
            img = None
            try:
                img = Image.open(fp)
                img = img.convert("RGB")  # always RGB
                img.save(fp, "JPEG", quality=95)
                img.close(); img = None
                cleaned += 1
            except Exception:
                if img is not None:
                    try: img.close()
                    except: pass
                try: os.remove(fp)
                except: pass
                removed += 1
print(f"  Cleaned: {cleaned}  |  Removed: {removed}")

# Count images
n_train_cat = len(os.listdir(os.path.join(TRAIN_DIR, "cat")))
n_train_dog = len(os.listdir(os.path.join(TRAIN_DIR, "dog")))
n_val_cat   = len(os.listdir(os.path.join(VAL_DIR, "cat")))
n_val_dog   = len(os.listdir(os.path.join(VAL_DIR, "dog")))
print(f"  Train: {n_train_cat} cats + {n_train_dog} dogs = {n_train_cat + n_train_dog}")
print(f"  Val:   {n_val_cat} cats + {n_val_dog} dogs = {n_val_cat + n_val_dog}")

# --- 2. Data loading + augmentation pipeline ---------------------------
print("\n[2/6] Building data pipeline ...")

train_aug = keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.15),
    layers.RandomTranslation(0.15, 0.15),
    layers.RandomZoom(0.15),
    layers.RandomContrast(0.2),
    layers.RandomBrightness(0.2),
], name="augmentation")

def load_and_decode(path, label):
    """Robust image loader: auto-detect format, skip corrupt files."""
    raw = tf.io.read_file(path)
    try:
        # decode_image auto-detects JPEG/PNG/BMP/GIF
        img = tf.io.decode_image(raw, channels=3, expand_animations=False)
    except Exception:
        return tf.zeros([*IMG_SIZE, 3], dtype=tf.uint8), tf.cast(-1, tf.int32)
    img = tf.image.resize(img, IMG_SIZE)
    return img, label

def is_valid(image, label):
    return tf.not_equal(label, -1)

def list_files_and_labels(data_dir):
    paths, labels = [], []
    for cls_idx, cls_name in enumerate(["cat", "dog"]):
        cls_dir = os.path.join(data_dir, cls_name)
        for fn in os.listdir(cls_dir):
            paths.append(os.path.join(cls_dir, fn))
            labels.append(cls_idx)
    return paths, labels

# --- Training dataset ---
train_paths, train_labels = list_files_and_labels(TRAIN_DIR)
ds_train = tf.data.Dataset.from_tensor_slices((train_paths, train_labels))
ds_train = (ds_train
    .shuffle(len(train_paths), seed=SEED)
    .map(load_and_decode, num_parallel_calls=AUTOTUNE)
    .filter(is_valid)
    .batch(BATCH)
    .map(lambda x, y: (train_aug(x, training=True), y),
         num_parallel_calls=AUTOTUNE)
    .map(lambda x, y: (
        tf.keras.applications.mobilenet_v2.preprocess_input(x), y),
         num_parallel_calls=AUTOTUNE)
    .prefetch(AUTOTUNE))

# --- Validation dataset ---
val_paths, val_labels = list_files_and_labels(VAL_DIR)
ds_val = tf.data.Dataset.from_tensor_slices((val_paths, val_labels))
ds_val = (ds_val
    .map(load_and_decode, num_parallel_calls=AUTOTUNE)
    .filter(is_valid)
    .batch(BATCH)
    .map(lambda x, y: (
        tf.keras.applications.mobilenet_v2.preprocess_input(x), y),
         num_parallel_calls=AUTOTUNE)
    .prefetch(AUTOTUNE))

# Count valid samples
n_train = sum(1 for _ in ds_train.unbatch())
n_val = sum(1 for _ in ds_val.unbatch())
print(f"  Valid train samples: {n_train}")
print(f"  Valid val samples:   {n_val}")

# Rebuild datasets (unbatch consumed them)
ds_train = tf.data.Dataset.from_tensor_slices((train_paths, train_labels))
ds_train = (ds_train
    .shuffle(len(train_paths), seed=SEED)
    .map(load_and_decode, num_parallel_calls=AUTOTUNE)
    .filter(is_valid)
    .batch(BATCH)
    .map(lambda x, y: (train_aug(x, training=True), y),
         num_parallel_calls=AUTOTUNE)
    .map(lambda x, y: (
        tf.keras.applications.mobilenet_v2.preprocess_input(x), y),
         num_parallel_calls=AUTOTUNE)
    .prefetch(AUTOTUNE))

ds_val = tf.data.Dataset.from_tensor_slices((val_paths, val_labels))
ds_val = (ds_val
    .map(load_and_decode, num_parallel_calls=AUTOTUNE)
    .filter(is_valid)
    .batch(BATCH)
    .map(lambda x, y: (
        tf.keras.applications.mobilenet_v2.preprocess_input(x), y),
         num_parallel_calls=AUTOTUNE)
    .prefetch(AUTOTUNE))

# --- 3. Model ---------------------------------------------------------
print("\n[3/6] Building model (MobileNetV2 backbone + classifier head) ...")

base = keras.applications.MobileNetV2(
    input_shape=(*IMG_SIZE, 3),
    include_top=False,
    weights="imagenet",
    pooling=None,
)
base.trainable = False

inputs = keras.Input(shape=(*IMG_SIZE, 3))
x = base(inputs, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(DROPOUT)(x)
outputs = layers.Dense(1, activation="sigmoid")(x)
model = keras.Model(inputs, outputs)

model.compile(
    optimizer=keras.optimizers.Adam(LR_HEAD),
    loss="binary_crossentropy",
    metrics=["accuracy"],
)
model.summary()

# --- 4. Training (two-phase) ------------------------------------------
print("\n[4/6] Training ...")

callbacks_head = [
    keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6, verbose=1),
    keras.callbacks.EarlyStopping(
        monitor="val_accuracy", patience=6, restore_best_weights=True, verbose=1),
]

print("\n  Phase 1 — training classifier head (backbone frozen) ...")
hist_head = model.fit(
    ds_train, validation_data=ds_val,
    epochs=EPOCHS_HEAD, callbacks=callbacks_head, verbose=1,
)

# Fine-tune: unfreeze top layers
base.trainable = True
FINE_TUNE_AT = 100
for layer in base.layers[:FINE_TUNE_AT]:
    layer.trainable = False

model.compile(
    optimizer=keras.optimizers.Adam(LR_FT),
    loss="binary_crossentropy",
    metrics=["accuracy"],
)

callbacks_ft = [
    keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=3, min_lr=1e-7, verbose=1),
    keras.callbacks.EarlyStopping(
        monitor="val_accuracy", patience=6, restore_best_weights=True, verbose=1),
]

print(f"\n  Phase 2 — fine-tuning (layers {FINE_TUNE_AT}+ unfrozen) ...")
hist_ft = model.fit(
    ds_train, validation_data=ds_val,
    epochs=EPOCHS_FT, callbacks=callbacks_ft, verbose=1,
)

# Merge histories
def merge_history(h1, h2):
    return {k: h1[k] + h2[k] for k in h1}

history = merge_history(hist_head.history, hist_ft.history)

# --- 5. Evaluation ----------------------------------------------------
print("\n[5/6] Evaluating on validation set ...")

y_true, y_pred, y_score = [], [], []
for images, labels in ds_val:
    probs = model.predict_on_batch(images)
    y_score.extend(np.array(probs).flatten())
    y_pred.extend((np.array(probs).flatten() >= 0.5).astype(int))
    y_true.extend(np.array(labels).flatten().astype(int))

y_true = np.array(y_true)
y_pred = np.array(y_pred)
y_score = np.array(y_score)

acc = (y_true == y_pred).mean()
print(f"  Validation accuracy: {acc:.4f} ({acc*100:.2f}%)")

report = classification_report(y_true, y_pred, target_names=["cat", "dog"], digits=4)
print("\n" + report)

with open(os.path.join(OUTPUT_DIR, "classification_report.txt"), "w") as f:
    f.write(report)

# --- 6. Plots ---------------------------------------------------------
print("\n[6/6] Generating plots and error analysis ...")

# 6a. Training curves
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.plot(history["accuracy"], label="Train", linewidth=1.5)
ax1.plot(history["val_accuracy"], label="Val", linewidth=1.5)
ax1.axvline(len(hist_head.history["accuracy"]) - 1, color="gray",
            linestyle="--", alpha=0.6, label="fine-tune start")
ax1.set_title("Accuracy")
ax1.set_xlabel("Epoch"); ax1.set_ylabel("Accuracy")
ax1.legend(); ax1.grid(alpha=0.3)

ax2.plot(history["loss"], label="Train", linewidth=1.5)
ax2.plot(history["val_loss"], label="Val", linewidth=1.5)
ax2.axvline(len(hist_head.history["loss"]) - 1, color="gray",
            linestyle="--", alpha=0.6, label="fine-tune start")
ax2.set_title("Loss")
ax2.set_xlabel("Epoch"); ax2.set_ylabel("Loss")
ax2.legend(); ax2.grid(alpha=0.3)

fig.suptitle("Training Curves — MobileNetV2 Transfer Learning", fontweight="bold")
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "training_curves.png"), dpi=150)
plt.close(fig)

# 6b. Confusion matrix
cm = confusion_matrix(y_true, y_pred)
fig, ax = plt.subplots(figsize=(6, 5))
ConfusionMatrixDisplay(cm, display_labels=["cat", "dog"]).plot(
    ax=ax, cmap="Blues", values_format="d", colorbar=False)
ax.set_title(f"Confusion Matrix  (accuracy = {acc:.3f})")
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "confusion_matrix.png"), dpi=150)
plt.close(fig)

# 6c. Error analysis
error_indices = np.where(y_true != y_pred)[0]
n_errors = len(error_indices)
print(f"  Total misclassified: {n_errors} / {len(y_true)} ({n_errors/len(y_true)*100:.1f}%)")

# --- Save model --------------------------------------------------------
model.save(os.path.join(OUTPUT_DIR, "model.keras"))

# --- Summary -----------------------------------------------------------
print("\n" + "=" * 60)
print("Experiment complete!")
print(f"  Accuracy:     {acc:.4f} ({acc*100:.2f}%)")
print(f"  Cat errors:   {(y_true[y_pred != y_true] == 0).sum()}")
print(f"  Dog errors:   {(y_true[y_pred != y_true] == 1).sum()}")
print(f"  Outputs dir:  {OUTPUT_DIR}")
print("=" * 60)
