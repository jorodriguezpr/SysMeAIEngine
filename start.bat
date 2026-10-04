@echo off
REM SysMeAIEngine - Private AI Model Engine
REM Copyright 2026 Jose Rodriguez
REM
REM Developer: Jose Rodriguez <jrpcone@gmail.com>
REM GitHub:    https://github.com/jorodriguezpr
REM Site:      https://microrepair.net
REM
REM Licensed under the Apache License, Version 2.0 (the "License");
REM you may not use this file except in compliance with the License.
REM You may obtain a copy of the License at
REM
REM     http://www.apache.org/licenses/LICENSE-2.0
REM
REM Unless required by applicable law or agreed to in writing, software
REM distributed under the License is distributed on an "AS IS" BASIS,
REM WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
REM See the License for the specific language governing permissions and
REM limitations under the License.
REM
REM SPDX-License-Identifier: Apache-2.0

REM Double-click to start SysMeAIEngine and open its web UI in your browser.
REM (start_server.bat does the same without opening a browser -- use that
REM one for scheduled tasks or cluster nodes.)
setlocal
cd /d "%~dp0"
title SysMeAIEngine

if not exist ".venv\Scripts\python.exe" (
    echo SysMeAIEngine is not installed yet. Run install.bat first.
    pause
    exit /b 1
)

set "PORT=11534"
for /f "tokens=1,* delims==" %%a in ('findstr /b /c:"ENGINE_PORT=" .env 2^>nul') do set "PORT=%%b"

echo Starting SysMeAIEngine on http://127.0.0.1:%PORT%/ui
echo   Your browser opens in about 10 seconds.
echo   Leave this window open -- closing it stops the engine. Press Ctrl+C to stop.
echo.
start "" /min powershell -NoProfile -WindowStyle Hidden -Command "Start-Sleep -Seconds 10; Start-Process 'http://127.0.0.1:%PORT%/ui'"
".venv\Scripts\python.exe" -m src.cli engine serve

pause
endlocal
