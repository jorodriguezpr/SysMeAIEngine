# Distributed multi-node training

Fan a single fine-tuning job out across this machine (the master, rank 0)
and one or more other SysMeAIEngine instances (slave nodes, rank 1..N),
training one model together via PyTorch's `DistributedDataParallel` (DDP).

There is no separate "agent build" — every instance runs the exact same
app and can act as master, slave, or both. A node only ever receives a job
when some other instance has registered it (`cluster node-add`) and picked
it for a run.

## Installing on each node

Run **`scripts/setup_node.ps1`** on every machine that will participate
(master included) — it checks Python, creates/updates the venv, installs
`requirements.txt` (`-Amd` also installs `requirements-amd.txt`), verifies
the install actually imports and reports what device it detected, runs the
test suite, and (with `-Cluster`) sets `ENGINE_HOST=0.0.0.0` and generates
an `ENGINE_API_KEY` if one isn't already set, printing the exact
`cluster node-add` command to run on the master afterward:

```powershell
# on every participating machine, including the master:
.\scripts\setup_node.ps1 -Cluster -OpenFirewall
```

`-OpenFirewall` needs an elevated ("Run as Administrator") PowerShell;
without it, the script still runs, just skips that step and tells you so
(open TCP `11534` and TCP `29500-29550` inbound manually instead).

**Windows 10**: yes — now confirmed on a real Windows 10 machine (build
19045, an Intel i7-4712MQ/Haswell workstation), not just reasoned about.
It did hit a real install failure there, but not an OS-version issue: torch
failed to import with `OSError: [WinError 1114] ... c10.dll` — root-caused
via Windows Event Viewer (Application log, APPCRASH entry) to an outdated
system-wide `msvcp140.dll` (14.29.30037.0, VC++ 2015-2019 era) crashing
with an access violation the moment torch called into it. "A VC++
Redistributable is already installed" was not sufficient evidence — that
old DLL was already present, just outdated. Installing the *current*
redistributable (https://aka.ms/vs/17/release/vc_redist.x64.exe, which
upgrades the same file in place) fixed it immediately; confirmed via a
clean `import torch` afterward. This is now baked into
`setup_node.ps1`'s own error handling — it detects this exact failure
signature and prints the fix directly instead of a raw traceback. Ruled
out along the way, with real tests rather than assumptions, before landing
on the actual cause: a corrupted install (fresh reinstall in a brand-new
throwaway venv, still failed identically), the project's own venv/PATH
(bare `torch`-only venv in an unrelated folder, still failed identically),
and CPU AVX2 support (Haswell was the first Intel generation to support
it, so this CPU was never actually the problem).

**Live-verified while building this script** (against the real project
`.env`, not a throwaway one): the setup script's "run the test suite to
verify" step initially produced a wall of 13 unrelated-looking test
failures the first time it was actually run — not because anything was
broken, but because `-Cluster` had just generated a real `ENGINE_API_KEY`,
and the test suite's requests (reasonably) don't send one. A first fix
attempt (`$env:ENGINE_API_KEY = ""` before invoking pytest) looked correct
but *still* failed the same way — Windows has a real quirk where setting a
PowerShell `$env:` variable to an empty string removes it from a *child*
process's environment entirely rather than passing it through empty, so
`python-dotenv`'s "don't override an already-set variable" check saw the
key as absent and loaded the real one from the file anyway. The actual fix
blanks the `ENGINE_API_KEY` line in the `.env` *file* itself for the
duration of the test run (restored via `try`/`finally` immediately after) —
verified this restores the original file correctly and the test suite
passes clean (38/38) once the real key isn't in play.

## Why Gloo, not NCCL, and why every rank uses float32

NCCL (the fast backend) requires **every** rank to have an NVIDIA GPU — it
cannot include a CPU-only rank at all. Since the master here has no CUDA GPU
by default, this project always uses the **Gloo** backend instead, which is
slower than NCCL but is the only one that works across a CPU+CUDA-mixed
group.

DDP's gradient sync averages raw tensor bytes across ranks, so every rank's
trainable parameters must share one dtype. Mixing fp16-on-GPU-ranks with
fp32-on-the-CPU-rank is exactly the kind of cross-rank inconsistency that
produces obscure, hard-to-debug distributed failures — so distributed jobs
always force `float32` and disable fp16/AMP everywhere, regardless of what
any individual rank's own hardware could otherwise use. This sacrifices some
GPU-side speed for correctness on a mixed cluster. (Plain single-node CUDA
inference/training, outside the cluster feature, does still use fp16 — see
`src/core/device_select.py`.)

## The straggler problem — read this before you're confused by slow training

Synchronous DDP means **every step waits for the slowest rank**. If the
CPU-only master participates as a compute rank (its default choice in this
project) alongside GPU slave nodes, the whole job runs no faster than the
CPU rank's own step time — this can fully negate the benefit of the GPU
nodes. This is a known, real limitation of naive heterogeneous DDP, not a
bug. If this turns out to matter for you in practice, the fix is to exclude
the CPU node from `--nodes`/the node checkboxes for a given run so only
GPU-having ranks participate — the master can still orchestrate without
itself being rank 0's compute; that path isn't wired up as a toggle yet,
but excluding the master is as simple as not adding an --nodes flag that
routes rank 0 there (currently the master is always rank 0 — see "Known gaps"
below if you want this changed).

## Setup

Each participating node needs:

1. **Network reachability on two ports**, not one:
   - The normal HTTP API port (default 11534) — used for `/cluster/*` job
     submission, status, and file transfer.
   - A **separate rendezvous port** (default 29500, auto-picked from a small
     range if taken) — used directly by `torch.distributed`, not HTTP. This
     needs to be open specifically on the **master** — the direction is
     reversed from the HTTP calls above: each slave's `torch.distributed`
     process connects *into* the master's machine on this port, not the
     other way around.

   Open both in Windows Firewall (inbound, TCP) on every node —
   `setup_node.ps1 -OpenFirewall` does this. **Confirmed live gotcha**: a
   firewall rule can exist and show `Enabled: True` while still doing
   nothing, if the active network is classified as "Public" rather than
   "Private" (`Get-NetConnectionProfile` shows which) — the rule above is
   scoped to the Private profile only, and Windows classifies a lot of home
   Wi-Fi networks as Public by default. `setup_node.ps1 -OpenFirewall` now
   detects and warns about this specific case; the fix (as Administrator):
   ```powershell
   Set-NetConnectionProfile -InterfaceAlias "Wi-Fi" -NetworkCategory Private
   ```

2. **`ENGINE_HOST=0.0.0.0`** in `.env` on every node that should be
   reachable — the default `127.0.0.1` (see the Security section of the
   main README) only accepts local connections, which would make a slave
   node invisible to the master entirely.

3. **A shared API key is recommended**: set the same `ENGINE_API_KEY` on
   every node, and pass it as `--api-key` when registering each node on the
   master (`cluster node-add`). Without it, `/cluster/*` accepts unauthenticated
   requests from anyone who can reach the port — fine on a trusted home LAN,
   not otherwise.

4. **Matching (or close) `torch`/`transformers` versions** across all nodes.
   A version mismatch between ranks in the same process group can cause
   obscure serialization/protocol errors rather than a clear message —
   `cluster node-ping`/the orchestrator's preflight check surfaces each
   node's versions so you can compare before launching.

5. **The base model must be reachable on every node**: either an HF hub id
   (each node downloads it independently — needs internet on every node), or
   a local checkpoint directory, which the master zips and uploads to each
   slave automatically as part of job submission.

6. **Disable sleep on every node while it's in the cluster.** Confirmed
   live: a laptop going to sleep mid-job doesn't fail cleanly — the master
   loses HTTP connectivity to it entirely (a plain connection timeout on
   `/cluster/*`, not a clear "node went to sleep" error) and the in-progress
   training on both ranks just stalls forever with no error either, since
   synchronous DDP has no way to detect a peer that's actually suspended
   vs. one that's just being slow. Windows: Settings > System > Power &
   battery > Screen and sleep > set both "sleep" options to "Never" while
   a laptop is actively part of a cluster job.

## Usage

Start the server on every participating machine first (master included) --
double-click `start_server.bat` in the repo root (or `start_slave.bat` in
`gitdeploy/client/` on a slave laptop set up from that deploy package), or
run `.venv\Scripts\python.exe -m src.cli engine serve` directly. Leave it
running; there's no wake-on-demand.

```bat
:: on the master, register each slave once
python -m src.cli cluster node-add --name laptop-a --host 192.168.1.50 --port 11534 --api-key <same key>
python -m src.cli cluster node-add --name laptop-b --host 192.168.1.51 --port 11534 --api-key <same key>
python -m src.cli cluster node-list
python -m src.cli cluster node-ping --name laptop-a

:: launch a distributed job — this machine is always rank 0
python -m src.cli cluster train --model Qwen/Qwen2.5-0.5B-Instruct --dataset ebooks.jsonl ^
  --nodes laptop-a laptop-b --method lora --epochs 1

:: check aggregate status (master + every remote node)
python -m src.cli cluster status --run-name <id printed above>

:: find/cancel a job directly on one node -- e.g. after the master-side
:: process that launched it died without ever calling cancel, leaving an
:: orphaned rank running with no id otherwise recoverable from the master
python -m src.cli cluster node-jobs --name laptop-a
python -m src.cli cluster node-cancel-job --name laptop-a --job-id <id from node-jobs>
```

### Train from Folder, distributed

`scripts/train_from_folder.py` (and the `/ui/train-folder` page) can distribute
its fine-tune step across registered nodes too, instead of being local-only:

```bat
python scripts\train_from_folder.py --source C:\docs --job-name my-docs ^
  --base-model Qwen/Qwen2.5-0.5B-Instruct --nodes laptop-a laptop-b
```

or check the node checkboxes shown on `/ui/train-folder` (only appears once
at least one node is registered). Extraction, chunking, dataset-build, and
the final RAG-index build always stay local on the master — only the
fine-tune step itself runs distributed, as a real multi-rank job via the
same `orchestrator.launch_distributed_job()` the plain `cluster train`
command above uses. The script pings every listed node first and aborts
before launching if any is unreachable, then polls until every rank
finishes (or `--cluster-timeout-minutes`, default 240, elapses) before
continuing on to register the model and build the RAG index — since
synchronous DDP means one dead/stalled rank stalls the rest with no error,
a failed or cancelled rank on either the master or any node stops the wait
immediately rather than sitting at the timeout.

Whenever the wait ends without every rank reaching "done" (a rank failure
or a timeout), the script cleans up every rank before exiting: it
terminates its own local rank-0 process (if still running) and calls each
node's `/cluster/jobs/{id}/cancel` for its remote rank. **Confirmed live
this was needed**: a real distributed job stalled during DDP rendezvous
(see the Gloo interface-confusion notes below) and hit the timeout, but
without this cleanup step the rank-0 process and the remote rank were both
found still running, unattended, *hours* after the script itself had
already exited and reported failure.

Not supported with `--base-model minilm`/`tinygpt` (TinyGPT is single-node
only — rejected with a clear error before anything is launched, same as
plain `cluster train`).

**Live-verified** (same-machine loopback: a real second SysMeAIEngine
instance on this machine, registered as a node pointing at its own port,
per the "If you're setting up a new node" loopback advice above): a real
two-rank distributed job via `train_from_folder.py --nodes` completed
successfully end to end — preflight, launch, polling to completion, model
registration, and RAG-index build — producing two real, separate
checkpoints (rank 0 at the registered model's path, rank 1 at
`models/checkpoints/cluster-<run_name>-rank1`). Real separate physical
machines were not re-tested for this specific entry point, but it's the
identical `orchestrator`/`runner` code path already verified across two
physical machines for plain `cluster train` (see "Honesty / verification
status" below), just invoked from a different caller.

Or use the **Cluster** page in the admin UI (`/ui/cluster`) for the same
flow with a form instead of flags.

## Honesty / verification status

Originally built and tested on a single AMD-only development machine (no
NVIDIA GPU, no second physical machine) — see
[[project_sysmeaiengine_cuda_support]] for the same caveat on plain CUDA.
Since then, real cross-machine testing happened against one actual Windows
10 laptop; see [[project_sysmeaiengine_windows10_vcredist_bug]] and
[[project_sysmeaiengine_cluster_distributed_training]] for the full
debugging history. What's actually been verified, live, versus what hasn't:

**Verified live (same-machine simulation):**
- Real multi-process `torch.distributed` rendezvous + `DistributedDataParallel`
  wrapping + gradient sync, confirmed by inspecting `trainer.model_wrapped`
  after training (2 and 3 local CPU processes on 127.0.0.1) —
  `scripts/verify_cluster_loopback.py`. This caught two real bugs during
  development: relying on Trainer's env-var auto-detection alone silently
  produced N independent single-process training runs with **no gradient
  sync and no error** (only a log warning) — fixed by explicitly calling
  `torch.distributed.init_process_group()` before constructing
  `TrainingArguments`, and by explicitly passing `use_cpu=True` on non-CUDA
  ranks (Accelerate's CPU-distributed detection requires it explicitly; it
  silently falls through to `DistributedType.NO` otherwise).
- The full HTTP job-submission path end to end (multipart dataset upload,
  subprocess launch with rendezvous env vars, job status/log tracking,
  cancellation) via `tests/test_cluster.py`, using a real WORLD_SIZE=1 job.

**Verified live (two real, separate physical machines — a Windows 11 master
with several extra virtual network adapters, and a real Windows 10 laptop
on the same home Wi-Fi):**
- Node registry CRUD, and a real node-to-node HTTP round trip over the
  actual LAN (add a node, ping it, see its live `/cluster/info` device
  report from a genuinely different machine — confirmed by hostname, CPU
  model, and Windows build all matching that specific laptop).
- A real firewall/networking setup cycle end to end, including two gotchas
  that would hit any new node: (1) a firewall rule can exist and show
  `Enabled: True` while doing nothing, because it's scoped to the "Private"
  network profile and Windows classified the actual Wi-Fi connection as
  "Public" by default — fix is `Set-NetConnectionProfile ... -NetworkCategory
  Private`, not a looser firewall rule; (2) the rendezvous port needs to be
  open specifically on the **master** (the direction is reversed from the
  job-submission HTTP calls). `setup_node.ps1 -OpenFirewall` now detects
  and warns about the Public-profile case.
- Real `torch.distributed` rendezvous succeeding across two separate
  physical machines over real Wi-Fi (confirmed via `netstat`/
  `Get-NetTCPConnection` showing genuine `ESTABLISHED` TCP connections
  between the two machines' real IPs, and both ranks' logs printing past
  `init_process_group()` — the exact print statement gated on it succeeding).
  This needed one more fix beyond the firewall/network-profile ones: the
  master machine had extra virtual network adapters (VirtualBox
  host-only, OpenVPN TAP) alongside its real Wi-Fi adapter, which confused
  Gloo's automatic interface selection — the rendezvous connection itself
  (the TCPStore) succeeded, but the actual Gloo backend construction
  silently hung with no error. Fix: set `GLOO_SOCKET_IFNAME` to the real
  adapter name (e.g. `Wi-Fi`) in the environment before launching, on
  whichever node has extra virtual adapters.

**NOT verified / known follow-ups from the real two-machine test:**
- **A real CUDA GPU actually participating as a DDP rank** — the slave
  laptop does have an NVIDIA GPU, but `pip install torch` on it resolved to
  a CPU-only build (`torch.__version__` showed `+cpu`) — confirmed this
  project's earlier claim that the standard PyPI wheel always includes CUDA
  support was **wrong**; fixed in the docs (see the CUDA section) but not
  yet re-tested with an actual CUDA build installed there.
- **A second, different interface-detection stall**, past the
  `GLOO_SOCKET_IFNAME` fix above: after both ranks successfully completed
  `init_process_group()`, model loading, and dataset tokenization (all
  confirmed in both logs), the job stalled again with zero network
  throughput (confirmed via `Get-NetAdapterStatistics`) somewhere inside
  `Trainer`/`Accelerate`'s CPU-distributed bootstrap internals
  (`get_cpu_distributed_information()` or similar) — plausibly the *same*
  multi-adapter confusion as before, but in a separate code path that
  doesn't respect `GLOO_SOCKET_IFNAME`. Not root-caused before the test
  session ended (the slave laptop went to sleep mid-wait, which — see the
  power-settings note above — silently kills the whole job without a clear
  error, so the exact original stall was never re-confirmed after fixing
  sleep). Worth retrying with sleep disabled before assuming this is still
  broken.
- **New data point on the interface-confusion family of stalls**: a real
  `train_from_folder.py --nodes` run against this same master machine
  hung at rendezvous with the master's rank-0 log showing exactly one
  line — `[c10d] The client socket has failed to connect to
  [host.docker.internal]:29500` — then nothing further. Confirmed
  **non-deterministic and only partially mitigated by `GLOO_SOCKET_IFNAME`**:
  across several repeated attempts with it set to the real Wi-Fi adapter,
  the job sometimes got past rendezvous fine and sometimes hung on this
  exact line again, once with a *different* phantom hostname
  (`JORODRIG-AI.lan` — this machine's own hostname via mDNS, not Docker).
  Uninstalling VirtualBox and OpenVPN entirely did not eliminate it either
  — a `vEthernet (WSL)` adapter was still present and is the more likely
  remaining suspect. Not root-caused; `GLOO_SOCKET_IFNAME` reduces but does
  not reliably prevent this.
- **Do not confuse the above hang with a separate, unrelated crash** that
  looks similar at first (a rank going "failed" during a `--nodes` run):
  if a rank actually loads the model and starts the training loop before
  dying, that's the CPU thread-oversubscription crash documented in
  `docs/TRAINING.md` ("STATUS_ACCESS_VIOLATION"), not a networking issue —
  confirmed by reproducing the identical crash in a plain local
  (non-distributed) run with no networking involved at all. Fixed as of
  this writing (`OMP_NUM_THREADS`/`MKL_NUM_THREADS` capped in both
  `finetune_model.py` and `distributed_finetune.py`); the rendezvous hang
  above is a genuinely separate, still-open issue.
- The straggler behavior described above, measured rather than reasoned
  about.
- The orchestrator's full multi-remote-node dispatch path with more than one
  *distinct* remote node (tested against the constituent pieces separately,
  and against a self-registered loopback node for the simple parts, plus
  now one real remote node, but not a true 3+ separate-process fan-out in
  one run).

If you're setting up a new node: start with
`cluster node-ping`, then `scripts/verify_cluster_loopback.py`
copied to run *between* two real nodes (adjust `MASTER_ADDR` to a real LAN
IP) before trusting a real training job to it, and please note anything that
turns out wrong so it can be fixed and documented here.

## Known gaps / possible future work

- One process (= one rank) per node — no multi-GPU-per-node support.
- No way yet to exclude the master from compute participation (see the
  straggler section above) short of not registering it as rank 0, which
  isn't currently an option since the master is hardcoded as rank 0.
- No automatic retry/resume if a node drops mid-job — a dropped rank kills
  the whole synchronous DDP group; you'd need to relaunch from the last
  epoch checkpoint.
- TinyGPT (`minilm`) is intentionally single-node only — it's a ~2M param
  toy model with no realistic payoff from distributing it.
- `cluster node-jobs`/`node-cancel-job` only work against a node running
  this same version of the code — confirmed live: calling `node-jobs`
  against a not-yet-updated node returns `405 Method Not Allowed`, not a
  clear "this node needs updating" message. Redeploy `gitdeploy/client` to
  that node (see its own `INSTALL.md`) and restart its `engine serve`
  before expecting these to work there.
