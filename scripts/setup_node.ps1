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
  Set up this machine as a SysMeAIEngine node (works for the master or any
  slave -- it's the same install either way, see docs/CLUSTER.md).

.DESCRIPTION
  Checks Python, creates/updates the venv, installs requirements, verifies
  the install actually imports and reports what device it detected, and
  (with -Cluster) prepares this node to join a distributed job: sets
  ENGINE_HOST=0.0.0.0 so other nodes can reach it (the default 127.0.0.1
  only accepts local connections) and generates an ENGINE_API_KEY if one
  isn't already set.

  Written for/tested against Windows PowerShell 5.1 (the default on both
  Windows 10 and 11) -- no PS7-only syntax (ternary, ??, etc).

.PARAMETER Amd
  Also install requirements-amd.txt (AMD DirectML/NPU extras).

.PARAMETER Cluster
  Prepare this node for distributed training (ENGINE_HOST=0.0.0.0 + API key).

.PARAMETER OpenFirewall
  Add inbound firewall rules for the HTTP port and the torch.distributed
  rendezvous port range (29500-29550). Requires an elevated (Run as
  Administrator) PowerShell -- the script will tell you if it isn't.

.EXAMPLE
  .\scripts\setup_node.ps1
  Plain single-node install/verify.

.EXAMPLE
  .\scripts\setup_node.ps1 -Cluster -OpenFirewall
  Set this laptop up as a cluster node (master or slave) with firewall rules.
#>
param(
    [switch]$Amd,
    [switch]$Cluster,
    [switch]$OpenFirewall,
    [int]$Port = 11534
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

function Write-Step($msg) { Write-Host "`n==> $msg" -ForegroundColor Cyan }
function Write-Warn2($msg) { Write-Host "WARNING: $msg" -ForegroundColor Yellow }
function Write-Fail($msg) { Write-Host "FAILED: $msg" -ForegroundColor Red }

Write-Step "OS / PowerShell"
$os = Get-CimInstance Win32_OperatingSystem
Write-Host "  $($os.Caption) ($($os.Version)), PowerShell $($PSVersionTable.PSVersion)"

# -- 1. Python ----------------------------------------------------------------
Write-Step "Checking Python"
$pythonCmd = $null
foreach ($cand in @("python", "py")) {
    try {
        & $cand --version 2>&1 | Out-Null
        if ($LASTEXITCODE -eq 0) { $pythonCmd = $cand; break }
    } catch {}
}
if (-not $pythonCmd) {
    Write-Fail "Python not found on PATH. Install Python 3.11+ from https://python.org/downloads/ (check 'Add to PATH' during install), then re-run this script."
    exit 1
}
$verOut = & $pythonCmd --version 2>&1
Write-Host "  Found: $verOut ($pythonCmd)"
if ($verOut -match "Python (\d+)\.(\d+)") {
    $maj = [int]$Matches[1]; $min = [int]$Matches[2]
    if ($maj -lt 3 -or ($maj -eq 3 -and $min -lt 11)) {
        Write-Fail "Python $($Matches[0]) found, but this project requires >=3.11 (see pyproject.toml). Install a newer Python and re-run."
        exit 1
    }
}

# -- 2. Virtual environment ---------------------------------------------------
Write-Step "Virtual environment"
$venvPython = Join-Path $RepoRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $venvPython)) {
    Write-Host "  Creating .venv ..."
    & $pythonCmd -m venv .venv
    if ($LASTEXITCODE -ne 0) { Write-Fail "venv creation failed"; exit 1 }
} else {
    Write-Host "  .venv already exists"
}

# -- 3. Dependencies -----------------------------------------------------------
Write-Step "Installing requirements (this can take a few minutes the first time)"
& $venvPython -m pip install --upgrade pip --quiet
& $venvPython -m pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) { Write-Fail "pip install -r requirements.txt failed"; exit 1 }
if ($Amd) {
    Write-Host "  -Amd passed: installing requirements-amd.txt"
    & $venvPython -m pip install -r requirements-amd.txt
    if ($LASTEXITCODE -ne 0) { Write-Warn2 "requirements-amd.txt install failed -- continuing without it" }
}

# -- 4. .env -------------------------------------------------------------------
Write-Step "Configuration (.env)"
$envPath = Join-Path $RepoRoot ".env"
if (-not (Test-Path $envPath)) {
    Copy-Item (Join-Path $RepoRoot ".env.example") $envPath
    Write-Host "  Created .env from .env.example"
} else {
    Write-Host "  .env already exists -- leaving your existing values, only adding what's missing below"
}

function Set-EnvVar($path, $key, $value, [switch]$OnlyIfEmpty) {
    # OnlyIfEmpty: for things like an API key, where an existing value must
    # never be silently replaced (other config may already depend on it).
    # Without it (the ENGINE_HOST case below): a loopback-only host breaks
    # cluster mode regardless of whether it was the default or a deliberate
    # customization, so -Cluster needs to actually override it, not skip it
    # because it happened to already be non-empty -- caught by testing this
    # against a real .env that already had ENGINE_HOST=127.0.0.1 set.
    $content = Get-Content $path -Raw
    $pattern = "(?m)^$key=.*$"
    if ($content -match $pattern) {
        if ($OnlyIfEmpty -and $content -match "(?m)^$key=\S+") {
            return
        }
        $content = $content -replace $pattern, "$key=$value"
    } else {
        $content += "`n$key=$value`n"
    }
    Set-Content -Path $path -Value $content -NoNewline
}

if ($Cluster) {
    Write-Host "  -Cluster passed: setting ENGINE_HOST=0.0.0.0 so other nodes can reach this one"
    Set-EnvVar $envPath "ENGINE_HOST" "0.0.0.0"

    $envContent = Get-Content $envPath -Raw
    if ($envContent -notmatch "(?m)^ENGINE_API_KEY=\S+") {
        $genKey = -join ((48..57) + (97..122) | Get-Random -Count 32 | ForEach-Object {[char]$_})
        Set-EnvVar $envPath "ENGINE_API_KEY" $genKey -OnlyIfEmpty
        Write-Host "  Generated ENGINE_API_KEY (also shown again at the end): $genKey"
    } else {
        Write-Host "  ENGINE_API_KEY already set -- reusing it"
    }
}

# A fresh install has an empty model registry, but .env's default model is
# the built-in minilm -- register it so Chat/Weights work before anything
# else is set up. "--path=" (not --path "") because Windows PowerShell 5.1
# drops empty-string arguments to native programs.
$registered = @()
try { $registered = @((& $venvPython -m src.cli model list 2>$null | Out-String | ConvertFrom-Json).models | ForEach-Object { $_.name }) } catch { }
if ($registered -notcontains "minilm") {
    & $venvPython -m src.cli model register --name minilm --provider minilm "--path=" | Out-Null
    if ($LASTEXITCODE -eq 0) { Write-Host "  Registered the built-in starter model (minilm)" }
    else { Write-Warn2 "Could not register the built-in minilm model -- run: .venv\Scripts\python.exe -m src.cli model register --name minilm --provider minilm --path=" }
}

# -- 5. Firewall (optional) ----------------------------------------------------
if ($OpenFirewall) {
    Write-Step "Firewall rules"
    $isAdmin = ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
    if (-not $isAdmin) {
        Write-Warn2 "Not running as Administrator -- skipping firewall rules. Re-run this script 'as Administrator' with -OpenFirewall, or add rules manually for TCP $Port and TCP 29500-29550 (inbound)."
    } else {
        try {
            New-NetFirewallRule -DisplayName "SysMeAIEngine HTTP" -Direction Inbound -Protocol TCP -LocalPort $Port -Action Allow -Profile Private -ErrorAction SilentlyContinue | Out-Null
            New-NetFirewallRule -DisplayName "SysMeAIEngine Cluster Rendezvous" -Direction Inbound -Protocol TCP -LocalPort 29500-29550 -Action Allow -Profile Private -ErrorAction SilentlyContinue | Out-Null
            Write-Host "  Added inbound rules for TCP $Port and TCP 29500-29550 (Private profile)"
        } catch {
            Write-Warn2 "Could not add firewall rules automatically: $_"
        }

        # These rules only apply to the "Private" profile -- but Windows
        # often classifies a home Wi-Fi network as "Public" by default
        # (confirmed live: the rule existed and showed Enabled=True, but
        # torch.distributed's rendezvous connection was still silently
        # dropped, because the active network's actual profile didn't
        # match what the rule was scoped to). Check for this exact trap
        # instead of leaving it to be rediscovered the hard way again.
        $publicProfiles = Get-NetConnectionProfile | Where-Object { $_.NetworkCategory -eq "Public" }
        if ($publicProfiles) {
            Write-Warn2 "Network '$($publicProfiles[0].InterfaceAlias)' is classified as 'Public' -- the firewall rules just added (Private profile only) will NOT apply to it, even though they exist and show as enabled. For a trusted home/office LAN, reclassify it:"
            Write-Host "    Set-NetConnectionProfile -InterfaceAlias `"$($publicProfiles[0].InterfaceAlias)`" -NetworkCategory Private"
            Write-Host "  Then re-check with: Get-NetConnectionProfile"
        }
    }
}

# -- 6. Verify -----------------------------------------------------------------
Write-Step "Verifying the install"
$verifyScript = @"
import sys
sys.path.insert(0, r'$RepoRoot')
import fastapi, transformers, torch
from src.core.device_select import cuda_info, device_report
print('fastapi', fastapi.__version__)
print('transformers', transformers.__version__)
print('torch', torch.__version__)
print('cuda_info', cuda_info())
"@
$verifyOutput = $verifyScript | & $venvPython - 2>&1
$verifyOutput | ForEach-Object { Write-Host $_ }
if ($LASTEXITCODE -ne 0) {
    $verifyText = $verifyOutput -join "`n"
    # A specific, very recognizable real-world failure (hit on a real Windows
    # 10 machine while building this): torch's compiled core DLLs (c10.dll
    # etc.) fail to load with WinError 1114 when the Microsoft Visual C++
    # Redistributable isn't installed -- common on a machine that's never
    # had Visual Studio, games, or other dev tools on it. This is NOT the
    # same failure as a CPU too old for AVX2 (that shows as WinError
    # -1073741795 / STATUS_ILLEGAL_INSTRUCTION, not a DLL init failure), so
    # only match the specific signature rather than guessing broadly.
    if ($verifyText -match "WinError 1114" -or $verifyText -match "c10\.dll" -or $verifyText -match "DLL initialization routine failed") {
        Write-Fail "torch's compiled DLLs failed to load (WinError 1114)."
        Write-Host ""
        Write-Host "  This means the Microsoft Visual C++ Redistributable (x64) on this" -ForegroundColor Yellow
        Write-Host "  machine is either missing OR an outdated version. Confirmed on a real" -ForegroundColor Yellow
        Write-Host "  Windows 10 machine: 'already installed' was NOT enough -- it had an" -ForegroundColor Yellow
        Write-Host "  old msvcp140.dll (14.29.30037.0, VC++ 2015-2019 era) in System32 that" -ForegroundColor Yellow
        Write-Host "  crashed with an access violation the moment torch called into it" -ForegroundColor Yellow
        Write-Host "  (visible in Event Viewer > Windows Logs > Application as an APPCRASH" -ForegroundColor Yellow
        Write-Host "  with 'Faulting module name: msvcp140.dll'). Installing the CURRENT" -ForegroundColor Yellow
        Write-Host "  redistributable upgrades that same file in place and fixed it. Fix:" -ForegroundColor Yellow
        Write-Host "    1. Download and run: https://aka.ms/vs/17/release/vc_redist.x64.exe"
        Write-Host "       (always installs whatever is currently latest -- run it even if"
        Write-Host "       you believe one is already installed, it upgrades in place)"
        Write-Host "    2. Reboot"
        Write-Host "    3. Re-run this script"
        Write-Host "  If that alone doesn't fix it, check Event Viewer > Windows Logs >"
        Write-Host "  Application for an APPCRASH entry right after reproducing the error --"
        Write-Host "  the 'Faulting module name' there is the real next clue."
        exit 1
    }
    if ($verifyText -match "-1073741795" -or $verifyText -match "illegal instruction") {
        Write-Fail "torch crashed with an illegal-instruction error -- this CPU likely doesn't support AVX2, which current torch/transformers builds require."
        Write-Host "  There's no simple flag to make this project work without AVX2 -- an older, pre-AVX2 torch build would be needed, which is a bigger change than this script handles. See docs/CLUSTER.md or ask before proceeding on this machine." -ForegroundColor Yellow
        exit 1
    }
    Write-Fail "Install verification failed -- see error above."
    exit 1
}

# Confirmed live: a plain `pip install torch` can resolve to a CPU-only
# build ("+cpu" in torch.__version__) even on a machine with a real NVIDIA
# GPU -- this project used to (wrongly) claim that never happens. Detect
# that specific mismatch here instead of leaving it to be noticed later via
# Task Manager showing 0% GPU usage during a real job.
$verifyText = $verifyOutput -join "`n"
$hasNvidiaGpu = $false
try {
    $hasNvidiaGpu = [bool](Get-CimInstance Win32_VideoController -ErrorAction SilentlyContinue | Where-Object { $_.Name -match "NVIDIA" })
} catch {}
if ($hasNvidiaGpu -and $verifyText -match "torch [\d.]+\+cpu") {
    Write-Warn2 "This machine has an NVIDIA GPU, but the installed torch is a CPU-ONLY build (version ends in '+cpu'). It will work, just without GPU acceleration. To actually use the GPU:"
    Write-Host "    .venv\Scripts\python.exe -m pip uninstall torch -y"
    Write-Host "    .venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cu121"
    Write-Host "  (adjust the cu121 tag to match this GPU's driver if needed -- see pytorch.org)"
    Write-Host "  Then re-check: .venv\Scripts\python.exe -c `"import torch; print(torch.__version__)`" -- should NOT end in +cpu."
}

Write-Host "`nRunning the test suite (fast, offline)..." -ForegroundColor Cyan
# The test suite assumes no API key is configured (its requests send none) --
# without neutralizing this, a real ENGINE_API_KEY sitting in .env (either
# pre-existing or just generated above by -Cluster) leaks in via dotenv and
# every authenticated-route test fails with 401, which looks like a broken
# install but isn't.
#
# First attempt was `$env:ENGINE_API_KEY = ""` before invoking pytest --
# looked right, still failed the exact same way when actually run against a
# real .env with a real key. Root cause: on Windows, setting a PowerShell
# `$env:` var to an empty string does not give the *child* process an
# empty-valued variable, it removes the variable from that child's
# environment block entirely (a real, documented Win32 CreateProcess
# quirk) -- so dotenv's `override=False` check saw the key as absent, not
# empty, and loaded the real value from the file anyway. Blanking the
# ENGINE_API_KEY line in the .env *file* itself for the duration of the
# test run sidesteps that: dotenv sets it to "" inside the already-running
# pytest process (a plain in-process os.environ assignment, not a new
# process's inherited environment block), which reads back correctly.
$envBackup = Get-Content $envPath -Raw
try {
    Set-Content -Path $envPath -Value ($envBackup -replace "(?m)^ENGINE_API_KEY=.*$", "ENGINE_API_KEY=") -NoNewline
    & $venvPython -m pytest tests\ -q
    $testExit = $LASTEXITCODE
} finally {
    Set-Content -Path $envPath -Value $envBackup -NoNewline
}
if ($testExit -ne 0) {
    Write-Warn2 "Some tests failed -- the install may still be usable, but check the output above before trusting it in a real job."
} else {
    Write-Host "All tests passed." -ForegroundColor Green
}

# -- 7. Summary -----------------------------------------------------------------
Write-Step "Done"
$startBat = if (Test-Path (Join-Path $RepoRoot "start_slave.bat")) { "start_slave.bat" } else { "start_server.bat" }
if (Test-Path (Join-Path $RepoRoot $startBat)) {
    Write-Host "  Start the engine:  double-click $startBat (or: .venv\Scripts\python.exe -m src.cli engine serve)"
} else {
    Write-Host "  Start the engine:  .venv\Scripts\python.exe -m src.cli engine serve"
}
Write-Host "  Dashboard:         http://localhost:$Port/ui"
if ($Cluster) {
    $envContent = Get-Content $envPath -Raw
    $keyLine = ($envContent -split "`n" | Where-Object { $_ -match "^ENGINE_API_KEY=" } | Select-Object -First 1)
    $apiKeyValue = $keyLine -replace "^ENGINE_API_KEY=", ""
    Write-Host "`n  This node is set up for cluster mode."
    Write-Host "  On the MASTER, register this node with:"
    Write-Host "    python -m src.cli cluster node-add --name <pick-a-name> --host <this-machine's-LAN-IP> --port $Port --api-key $apiKeyValue"
    Write-Host "  Find this machine's LAN IP with: ipconfig"
    Write-Host "  See docs\CLUSTER.md for the full setup checklist (this script only opens the Private firewall profile -- adjust if this LAN is on a different profile)."
}
