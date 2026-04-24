$venvPython = "$PSScriptRoot\.venv\Scripts\python.exe"

Write-Host "Starting AegisDesk on http://localhost:7860 ..."

& $venvPython -m uvicorn server.app:app --host 0.0.0.0 --port 7860
