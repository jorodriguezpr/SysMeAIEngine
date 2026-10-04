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

REM Double-click this to install/verify SysMeAIEngine on this laptop and
REM prepare it as a cluster node. Right-click -> "Run as administrator" to
REM also open the Windows Firewall ports (recommended). See INSTALL.md.
setlocal
cd /d "%~dp0"
REM Works from the package root or from the repository's cluster-client folder
if not exist "scripts\setup_node.ps1" cd ..
powershell -NoProfile -ExecutionPolicy Bypass -File "scripts\setup_node.ps1" -Cluster -OpenFirewall
echo.
pause
