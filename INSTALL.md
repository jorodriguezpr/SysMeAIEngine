# Installing SysMeAIEngine

SysMeAIEngine runs on your own Windows PC. It reads your documents (PDF, Word,
PowerPoint, Excel, text, scanned pages), builds a searchable index, and answers
questions about them in a web page, with sources, without sending anything to
the cloud.

Developer: **Jose Rodriguez** · <jrpcone@gmail.com> ·
[github.com/jorodriguezpr](https://github.com/jorodriguezpr) ·
[microrepair.net](https://microrepair.net)

---

## 1. What you need

| | Minimum | Recommended |
|---|---|---|
| Windows | 10 or 11, 64-bit | Windows 11 |
| Memory (RAM) | 8 GB (uses the smaller 3B answering model) | 16 GB or more (7B answering model) |
| Free disk | 10 GB | 20 GB |
| CPU | Any 64-bit CPU with AVX2 (most CPUs from 2015 on) | 8+ cores |
| Python | 3.11 or 3.12 | 3.12 |
| Internet | Only during installation (about 5-8 GB of downloads) | |

No graphics card is needed; everything runs on the CPU.

**Install Python first** if you don't have it: download it from
<https://www.python.org/downloads/windows/> and, on the first installer screen,
tick **"Add python.exe to PATH"**.

## 2. Get SysMeAIEngine

- **ZIP:** on the GitHub page click **Code → Download ZIP**, then right-click
  the ZIP → **Extract All…** to a folder such as `C:\SysMeAIEngine`.
- **Git:** `git clone <repository URL> C:\SysMeAIEngine`

Avoid folders that sync to the cloud (OneDrive, Dropbox); the engine writes a
lot of small files while indexing.

## 3. Install (double-click)

Open the folder and double-click **`install.bat`**. It runs two steps:

1. **Python environment.** It creates a private Python environment
   (`.venv`), installs all requirements, checks your PC, and runs a quick
   self-test. **The first run can take 10-20 minutes** and may look stuck at
   "Installing requirements". Windows Defender scans every downloaded file,
   so let it finish.
2. **Answering model (recommended, answer Y).** It installs
   [Ollama](https://ollama.com) if needed and downloads the Qwen 2.5 Instruct
   model: 7B (~4.7 GB) on PCs with 14 GB+ RAM, otherwise 3B (~1.9 GB). It also
   switches Chat to *grounded* mode, so answers come from your documents.

At the end, answer **Y** to start the engine.

> Windows may show "Windows protected your PC" for `install.bat`. Click
> **More info → Run anyway**. The script only works inside its own folder
> (plus the Ollama install, if you accept it).

## 4. Start the engine

Double-click **`start.bat`**. A black window opens (keep it open; closing it
stops the engine) and your browser opens the web UI after about 10 seconds:

<http://127.0.0.1:11534/ui>

The engine only listens on your own PC (127.0.0.1) unless you change
`ENGINE_HOST` in `.env`.

## 5. Your first document assistant

Open the **Learn** tab in the web UI and follow **★ Start here**. In short:

1. **Train from Folder:** pick a folder of documents. To try it right away,
   use the included `examples\sample-docs` folder. Set Chunking to
   `paragraph` and click **Start**.
2. When the job is done, go to **Models**, set **RAG corpus** on your
   answering model (`qwen7b-local` or `qwen3b-local`) to the job's name, and
   click **Set**.
3. Ask questions in **Chat**. `examples\sample-docs\README.md` lists test
   questions with their answers.

## 6. Configuration (`.env`)

A ready-to-use **`.env`** with safe defaults is included. You don't have to
edit it. The most useful settings:

| Setting | Default | Meaning |
|---|---|---|
| `ENGINE_PORT` | 11534 | Web UI / API port (Ollama uses 11434, so they don't clash) |
| `ENGINE_HOST` | 127.0.0.1 | 0.0.0.0 lets other PCs on your network connect (set `ENGINE_API_KEY` too) |
| `ENGINE_API_KEY` | empty | When set, API calls must send this key |
| `ENGINE_UI_AUTH` | empty | `user:password` to protect the web UI |
| `ENGINE_DEFAULT_MODEL` | minilm (qwen7b-local / qwen3b-local after step 2) | Model used by Chat |
| `ENGINE_OCR` | auto | Read scanned pages when OCR is installed (see Troubleshooting) |

`.env.example` is a reference copy of all settings with comments. Restart the
engine after editing `.env`. Cloud AI providers (optional, for building
better training data) are configured on the **Settings** page; their keys are
stored in `models\cloud_settings.json`, never in `.env`.

## 7. Updating

Download the new version (or `git pull`) into the same folder and run
`install.bat` again. Your `data\` (documents, indexes) and `models\` folders
are kept, and so is your `.env`.

## 8. Uninstalling

Close the engine window, then delete the SysMeAIEngine folder. If you
installed Ollama only for this, remove it from **Settings → Apps**. Its models
are stored in `%USERPROFILE%\.ollama`.

## 9. Troubleshooting

| Problem | Fix |
|---|---|
| `python` is not recognized | Install Python 3.12 with **"Add python.exe to PATH"** ticked, then run `install.bat` again. |
| Installation stuck at "Installing requirements" | Normal on the first run (Defender scans every file). Wait up to 20-30 minutes. |
| Error mentioning `msvcp140.dll` or `torch` won't import | Install the latest Microsoft Visual C++ Redistributable: <https://aka.ms/vs/17/release/vc_redist.x64.exe>, reboot, run `install.bat` again. |
| "Ollama not found" | Install it from <https://ollama.com/download>, then run `scripts\setup_answer_model.ps1` (right-click → Run with PowerShell). |
| Browser says the page can't be reached | Wait until the black window shows `Uvicorn running on ...`, then refresh. Check the port in `.env`. |
| "Address already in use" / port busy | Another program uses the port. Set a different `ENGINE_PORT` in `.env`, e.g. 11634. |
| Chat ignores my documents | Grounded mode must be on (Settings), and the corpus must be set on the model (Models page). |
| Answers take 30+ seconds | Normal on CPU for the 7B model. The 3B model is faster but less accurate. |
| Scanned PDFs give no text | Install OCR: `.venv\Scripts\pip install -r requirements-ocr.txt onnxruntime`, then restart the engine (the OCR card on Train from Folder should say *on*). |

More detail: `README.md` (features, API, CLI), `docs\` (training, grounding,
cluster, MCP), and the **Learn** tab inside the app.

## Multi-PC cluster (advanced)

To spread training across several PCs, see `docs\CLUSTER.md`. Each extra PC
uses the `cluster-client` setup (`cluster-client\INSTALL.md`).
