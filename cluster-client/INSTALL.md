# SysMeAIEngine -- slave node install

This folder is a self-contained copy of everything needed to run this
laptop as a slave (or master) node in a SysMeAIEngine distributed training
cluster. It does not include any trained models or a configured `.env` --
those are either created fresh here or transferred automatically per-job by
the master (see `docs/CLUSTER.md`).

## Steps

1. Copy this whole `client` folder onto the laptop (USB drive, network
   share, zip -- whatever's convenient). Anywhere is fine, e.g.
   `C:\SysMeAIEngine\`.

2. Install Python 3.11 or newer if it isn't already there:
   https://www.python.org/downloads/windows/ -- check **"Add python.exe to
   PATH"** during install.

3. Double-click **`install.bat`** (or right-click `scripts\setup_node.ps1`
   -> **Run with PowerShell**, or run it from a PowerShell window yourself).
   This checks Python, creates a virtual environment, installs everything
   needed, verifies the install actually works, and prepares this node for
   cluster mode (`ENGINE_HOST=0.0.0.0` + a generated API key).

   To also open the Windows Firewall ports this needs (recommended so the
   master can actually reach it), right-click `install.bat` ->
   **Run as administrator** instead.

   **This can take 10-20+ minutes the first time**, mostly stuck looking
   like nothing is happening after "Installing requirements..." -- that's
   normal, not frozen. Windows Defender's real-time protection scans every
   file as `torch` (a very large package) gets extracted, which is by far
   the slowest part. Confirmed on this machine: pip itself finishes
   downloading in under a minute, then the actual file extraction/install
   step alone took about 15 minutes with Defender active. Just let it run.

4. When it finishes, it prints something like:

   ```
   On the MASTER, register this node with:
     python -m src.cli cluster node-add --name <pick-a-name> --host <this-machine's-LAN-IP> --port 11534 --api-key <a-generated-key>
   Find this machine's LAN IP with: ipconfig
   ```

   Run `ipconfig` on **this** laptop to get its actual LAN IP (something
   like `192.168.1.xx`), then go back to the **master** machine and run
   that `cluster node-add` command there, filling in the real IP.

5. This laptop's own SysMeAIEngine server needs to already be running
   whenever the master wants to reach it -- there's no wake-on-demand.
   Double-click **`start_slave.bat`** (in this same folder) and leave the
   window open -- it prints this laptop's LAN IP and starts the server.
   Under the hood that's just:

   ```
   .venv\Scripts\python.exe -m src.cli engine serve
   ```

   run it directly instead of the .bat if you'd rather. There's no
   automatic-start-on-boot wiring yet -- see the "Windows service
   installer" line in the main README's roadmap (not built yet; for now,
   just leave the window open, or use Task Scheduler to run `start_slave.bat`
   at login).

## If this laptop has an NVIDIA GPU

Nothing extra to install for CUDA -- the standard `torch` package already
installed by step 3 includes CUDA support and uses a real NVIDIA GPU
automatically when the driver is present. Confirm it was detected:

```
.venv\Scripts\python.exe -c "from src.core.device_select import cuda_info; print(cuda_info())"
```

If `available` comes back `false` on a machine you know has an NVIDIA card,
check the NVIDIA driver is installed (Device Manager -> Display adapters)
-- that's the only piece this install doesn't/can't handle for you.

## If this laptop has no GPU (CPU only)

That's fine -- it can still join as a rank, just note the whole distributed
job runs at the speed of its slowest rank (see "The straggler problem" in
`docs/CLUSTER.md`) if this node participates in the same job as GPU nodes.

## Troubleshooting

- **"Python not found"** -- installed it without checking "Add to PATH"?
  Reinstall and check that box, or add it to PATH manually.
- **`OSError: [WinError 1114] ... c10.dll`** during "Verifying the install"
  -- confirmed on a real Windows 10 machine: this means the Microsoft
  Visual C++ Redistributable is missing *or outdated* -- "one is already
  installed" is not enough to rule this out. In the confirmed case, Event
  Viewer (Windows Logs > Application) showed an APPCRASH with
  `Faulting module name: msvcp140.dll` at an old version
  (14.29.30037.0, from the VC++ 2015-2019 era) -- installing the *current*
  redistributable upgrades that same file in place and fixed it:
  1. Download and run https://aka.ms/vs/17/release/vc_redist.x64.exe (this
     always installs whatever is currently latest -- run it even if you
     believe one is already installed)
  2. Reboot
  3. Re-run `install.bat`
  If that alone doesn't fix it, reproduce the error and immediately check
  Event Viewer > Windows Logs > Application for an APPCRASH entry -- the
  "Faulting module name" there is the real next clue (this is also now
  built into `setup_node.ps1`'s own error message when it detects this
  failure).
- **Firewall / master can't reach this node** -- re-run `install.bat` "as
  administrator" (step 3), or manually allow inbound TCP `11534` and TCP
  `29500-29550` for this app.
- **Version mismatch errors during a real job** -- run
  `.venv\Scripts\python.exe -c "import torch, transformers; print(torch.__version__, transformers.__version__)"`
  here and on the master; they should match closely. Re-run
  `pip install -r requirements.txt --upgrade` on whichever is behind.
- Anything else: see `docs/CLUSTER.md`'s "Honesty / verification status"
  section for what's actually been tested versus not.
