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

REM Double-click to install SysMeAIEngine on this PC (see INSTALL.md):
REM   1. Python environment + all requirements + config check + self-test
REM   2. (optional) the local answering model: Ollama + Qwen 2.5 Instruct
REM   3. (optional) start the engine and open it in your browser
REM Safe to run again later: each step skips what is already done.
setlocal
cd /d "%~dp0"
title SysMeAIEngine installer

echo ==========================================================
echo   SysMeAIEngine installer
echo ==========================================================
echo.
echo Step 1 of 2: Python environment and requirements.
echo   The first run downloads about 2-3 GB and can take 10-20 minutes.
echo   It may look stuck at "Installing requirements" -- let it finish.
echo.
powershell -NoProfile -ExecutionPolicy Bypass -File "scripts\setup_node.ps1"
if errorlevel 1 (
    echo.
    echo Installation did not finish -- read the messages above, fix the problem, and run install.bat again.
    pause
    exit /b 1
)

echo.
echo ==========================================================
echo Step 2 of 2: local answering model (recommended)
echo   Installs Ollama if needed and downloads Qwen 2.5 Instruct
echo   (about 4.7 GB for 7B, 1.9 GB for 3B on PCs with less RAM).
echo   Without it, Chat only has the tiny built-in demo model.
echo ==========================================================
choice /C YN /M "Set up the local answering model now"
if errorlevel 2 goto skipmodel
powershell -NoProfile -ExecutionPolicy Bypass -File "scripts\setup_answer_model.ps1"
:skipmodel

echo.
echo Installation complete. Start the engine any time with start.bat.
choice /C YN /M "Start SysMeAIEngine now"
if errorlevel 2 goto end
call "%~dp0start.bat"
:end
endlocal
