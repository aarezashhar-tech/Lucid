@echo off
echo ========================================================
echo Starting Lucid AI News Verification Platform (Local)
echo ========================================================

:: Check Python
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [!] Python is not in PATH or not installed. Please install Python 3.10+ from python.org
    pause
    exit /b
)

:: Check Node
node --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [!] Node.js is not in PATH or not installed. Please install Node.js 18+ from nodejs.org
    pause
    exit /b
)

echo [*] Starting Backend API Server (Port 5000)...
start "Lucid Backend" cmd /k "cd backend && pip install -r requirements.txt && python app.py"

echo [*] Starting Frontend Vite Server (Port 5173)...
start "Lucid Frontend" cmd /k "cd frontend && npm install && npm run dev"

echo [OK] Both servers are launching.
echo Backend:  http://127.0.0.1:5000/api/health
echo Frontend: http://localhost:5173
echo ========================================================
pause
