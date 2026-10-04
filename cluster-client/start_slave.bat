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

REM Starts this laptop's SysMeAIEngine server so it can act as a cluster
REM slave (or master) node. The master can only reach this node while this
REM keeps running -- there's no wake-on-demand -- so leave this window
REM open for as long as this laptop is part of the cluster. See INSTALL.md.
setlocal
cd /d "%~dp0"
REM Works from the package root or from the repository's cluster-client folder
if not exist "scripts\setup_node.ps1" cd ..

if not exist ".venv\Scripts\python.exe" (
    echo ERROR: .venv not found in %~dp0
    echo Run install.bat first to set up this node.
    pause
    exit /b 1
)

echo This laptop's LAN IP (give this to the master for "cluster node-add"):
ipconfig | findstr /R /C:"IPv4"
echo.
echo Reminders while this node is part of a cluster job:
echo   - Disable sleep: Settings ^> System ^> Power ^& battery ^> Screen and sleep.
echo   - Keep this window open the whole time -- closing it stops the server.
echo.
echo Starting SysMeAIEngine slave node...  (Press Ctrl+C to stop)
echo.
".venv\Scripts\python.exe" -m src.cli engine serve

pause
