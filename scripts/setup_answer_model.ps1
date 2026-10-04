# SysMeAIEngine - Private AI Model Engine
# Copyright 2026 Jose Rodriguez
#
# Developer: Jose Rodriguez <jrpcone@gmail.com>
# GitHub:    https://github.com/jorodriguezpr
# Site:      https://microrepair.net
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

<#
.SYNOPSIS
  Sets up the local answering model so Chat can answer from your documents
  with no cloud service: Ollama + Qwen 2.5 Instruct (7B, or 3B on low-RAM PCs).

.DESCRIPTION
  1. Finds Ollama, or offers to install it (winget), or tells you where to get it.
  2. Downloads the model with "ollama pull" (7B ~4.7 GB, 3B ~1.9 GB).
  3. Registers it in the engine (qwen7b-local / qwen3b-local) and makes it the
     default model in .env.
  4. Moves the engine off port 11434 (Ollama's own port) to 11534, for older .env files.
  5. Turns on grounded chat with the measured-best settings (top-3 passages,
     "complete" answer style).
  Safe to re-run: every step checks what is already done.

.PARAMETER Size
  7b (default; needs ~16 GB RAM for comfortable use) or 3b (8 GB PCs; lower accuracy).

.PARAMETER Yes
  Don't ask questions; accept the defaults (install Ollama with winget if missing).
#>
param(
    [ValidateSet("auto", "7b", "3b")][string]$Size = "auto",
    [switch]$Yes
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot
$venvPython = Join-Path $RepoRoot ".venv\Scripts\python.exe"
$envPath = Join-Path $RepoRoot ".env"

function Write-Step($msg) { Write-Host "`n==> $msg" -ForegroundColor Cyan }
function Write-Warn2($msg) { Write-Host "WARNING: $msg" -ForegroundColor Yellow }
function Write-Fail($msg) { Write-Host "FAILED: $msg" -ForegroundColor Red }
function Ask($question, $default = "Y") {
    if ($Yes) { return $true }
    $a = Read-Host "$question [$(if ($default -eq 'Y') { 'Y/n' } else { 'y/N' })]"
    if ([string]::IsNullOrWhiteSpace($a)) { $a = $default }
    return $a -match '^[Yy]'
}
function Set-EnvVar($key, $value) {
    $lines = @()
    if (Test-Path $envPath) { $lines = @(Get-Content $envPath) }
    $found = $false
    for ($i = 0; $i -lt $lines.Count; $i++) {
        if ($lines[$i] -match "^\s*$([regex]::Escape($key))\s*=") { $lines[$i] = "$key=$value"; $found = $true }
    }
    if (-not $found) { $lines += "$key=$value" }
    Set-Content -Path $envPath -Value $lines -Encoding ascii
}
function Get-EnvVar($key) {
    if (-not (Test-Path $envPath)) { return $null }
    foreach ($l in Get-Content $envPath) { if ($l -match "^\s*$([regex]::Escape($key))\s*=\s*(.*)$") { return $Matches[1].Trim() } }
    return $null
}

if (-not (Test-Path $venvPython)) {
    Write-Fail ".venv not found. Run install.bat (or scripts\setup_node.ps1) first."
    exit 1
}
if (-not (Test-Path $envPath)) { Copy-Item (Join-Path $RepoRoot ".env.example") $envPath }

# -- 1. model size from RAM -----------------------------------------------------
$ramGb = [math]::Round((Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory / 1GB)
if ($Size -eq "auto") { $Size = if ($ramGb -ge 14) { "7b" } else { "3b" } }
$tag = "qwen2.5:$Size-instruct"
$name = "qwen$Size-local"
Write-Step "Answering model: $tag (this PC has $ramGb GB RAM)"
if ($Size -eq "3b") { Write-Host "  3B is used on PCs under ~14 GB RAM. It answers fewer questions correctly than 7B (measured 36% vs 67%). Re-run with -Size 7b to override." }

# -- 2. Ollama ------------------------------------------------------------------
Write-Step "Ollama (runs the model on this PC)"
$ollama = Get-Command ollama -ErrorAction SilentlyContinue
if (-not $ollama) {
    $default = Join-Path $env:LOCALAPPDATA "Programs\Ollama\ollama.exe"
    if (Test-Path $default) { $ollama = Get-Item $default }
}
if (-not $ollama) {
    $winget = Get-Command winget -ErrorAction SilentlyContinue
    if ($winget -and (Ask "  Ollama is not installed. Install it now with winget?")) {
        & winget install --id Ollama.Ollama -e --accept-source-agreements --accept-package-agreements
        $default = Join-Path $env:LOCALAPPDATA "Programs\Ollama\ollama.exe"
        if (Test-Path $default) { $ollama = Get-Item $default }
    }
}
if (-not $ollama) {
    Write-Warn2 "Ollama not found. Install it from https://ollama.com/download , then re-run:"
    Write-Host  "    powershell -ExecutionPolicy Bypass -File scripts\setup_answer_model.ps1"
    exit 2
}
$ollamaExe = if ($ollama.Source) { $ollama.Source } else { $ollama.FullName }
Write-Host "  Found: $ollamaExe"

# make sure the Ollama service answers (the desktop app starts it; start it if not)
$up = $false
try { Invoke-RestMethod -Uri "http://127.0.0.1:11434/api/tags" -TimeoutSec 5 | Out-Null; $up = $true } catch { }
if (-not $up) {
    Write-Host "  Starting Ollama..."
    Start-Process -FilePath $ollamaExe -ArgumentList "serve" -WindowStyle Hidden
    for ($i = 0; $i -lt 20 -and -not $up; $i++) {
        Start-Sleep -Seconds 1
        try { Invoke-RestMethod -Uri "http://127.0.0.1:11434/api/tags" -TimeoutSec 3 | Out-Null; $up = $true } catch { }
    }
}
if (-not $up) { Write-Fail "Ollama is installed but not answering on http://127.0.0.1:11434 . Open the Ollama app once, then re-run."; exit 3 }

# -- 3. download the model -------------------------------------------------------
Write-Step "Downloading $tag (one time; 7B is ~4.7 GB, 3B ~1.9 GB)"
$have = (Invoke-RestMethod -Uri "http://127.0.0.1:11434/api/tags").models | Where-Object { $_.name -eq $tag -or $_.model -eq $tag }
if ($have) { Write-Host "  Already downloaded." }
else {
    & $ollamaExe pull $tag
    if ($LASTEXITCODE -ne 0) { Write-Fail "ollama pull $tag failed (check your internet connection and disk space)"; exit 4 }
}

# -- 4. register + make default ------------------------------------------------------
Write-Step "Registering '$name' in the engine and making it the default model"
$registered = @()
try { $registered = @((& $venvPython -m src.cli model list 2>$null | Out-String | ConvertFrom-Json).models | ForEach-Object { $_.name }) } catch { }
if ($registered -contains $name) { Write-Host "  '$name' is already registered." }
else {
    & $venvPython -m src.cli model register --name $name --provider ollama --path $tag
    if ($LASTEXITCODE -ne 0) { Write-Fail "model register failed"; exit 5 }
}
Set-EnvVar "ENGINE_DEFAULT_MODEL" $name
Write-Host "  .env: ENGINE_DEFAULT_MODEL=$name"

if ((Get-EnvVar "ENGINE_PORT") -eq "11434") {
    Set-EnvVar "ENGINE_PORT" "11534"
    Write-Host "  .env: ENGINE_PORT=11534 (11434 belongs to Ollama; the web UI is now http://127.0.0.1:11534/ui)"
}

# -- 5. grounded chat with the measured-best settings ------------------------------------
Write-Step "Turning on grounded chat (answers come from your documents, with sources)"
$py = @'
import types
from src.core.config import Config
from src.core.grounding_settings import save_grounding_settings
s = save_grounding_settings(types.SimpleNamespace(config=Config.from_env()),
                            chat_mode="grounded", chat_top_k=3, local_prompt_style="complete")
print("  chat_mode=%s, chat_top_k=%s, local_prompt_style=%s" % (s["chat_mode"], s["chat_top_k"], s["local_prompt_style"]))
'@
$py | & $venvPython -
if ($LASTEXITCODE -ne 0) { Write-Warn2 "Could not save grounding settings; set Grounded mode on the Settings page instead." }

$port = Get-EnvVar "ENGINE_PORT"
Write-Host "`nDone. Start the engine with start_server.bat, then open http://127.0.0.1:$port/ui" -ForegroundColor Green
Write-Host "New here? Open the Learn tab and follow 'Start here' (sample documents are in examples\sample-docs)."
