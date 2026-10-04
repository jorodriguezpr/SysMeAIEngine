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

REM Starts the SysMeAIEngine server on this machine. Works whether this
REM machine is standalone, a cluster master, or a cluster slave -- it's
REM the exact same server process either way (see docs\CLUSTER.md).
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo ERROR: .venv not found in %~dp0
    echo Run scripts\setup_node.ps1 first to create it and install dependencies.
    pause
    exit /b 1
)

echo Starting SysMeAIEngine...
echo   Leave this window open -- closing it stops the server.
echo   Press Ctrl+C to stop.
echo.
".venv\Scripts\python.exe" -m src.cli engine serve

pause
