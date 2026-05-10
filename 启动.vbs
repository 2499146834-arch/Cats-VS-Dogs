Set ws = CreateObject("Wscript.Shell")
' Start Flask server in background (hidden window)
ws.Run """D:\Qwen 2.5 7B\env\python.exe"" ""D:\Cats vs Dogs\app.py""", 0, False
' Wait for server to start then open browser
WScript.Sleep 3000
ws.Run "http://127.0.0.1:5001"
