# Architecture

```mermaid
flowchart LR
  subgraph Clients
    A1[Web UI]
    A2[OpenAI-compatible client]
    A3[Anthropic Messages API client]
    A4[MCP client]
  end

  subgraph SysMeAIEngine
    CLI[src/cli.py] -->|engine serve| Server[FastAPI server]
    Server --> API[api/*]
    Server --> Admin[admin/*]
    Server --> MCPEndpoint[MCP JSON-RPC]
    Server --> ModelLoader[core/model_loader.py]
    ModelLoader --> ProviderRegistry[core/providers/*]
    ProviderRegistry --> Min[MiniLM TinyGPT]
    ProviderRegistry --> HF[HF Transformers]
    ProviderRegistry --> GGUF[llama.cpp]
    ProviderRegistry --> DML[ONNX + DirectML]
    ProviderRegistry --> NPU[ONNX + Ryzen AI NPU]
    ModelLoader --> State[core/state.py]
    State --> Metrics[core/metrics.py] --> psutil/dxdiag/pnputil
    Admin --> Templates[web/templates/*]
  end

  subgraph Storage
    DataRaw[(data/raw)]
    DataChunks[(data/chunks)]
    Datasets[(data/datasets)]
    Ckpts[(models/checkpoints)]
    Weights[(models/weights)]
  end

  Scripts[scripts/*.py] --> DataRaw
  Scripts --> DataChunks
  Scripts --> Datasets
  Scripts --> Ckpts
  Scripts --> Weights

  Clients --> Server
  AgentFlow[agent_self_evolve.py] --> DataRaw
```

## Key invariants

- **`EngineState`** is a single per-process object: `loaded` model, rolling metrics, agent history, training jobs. Never import-time-singleton — explicit `init_state(cfg)` in `build_app()`.
- **Providers** return `(model, tokenizer)`; `llama.cpp` uses `model == tokenizer`. `minilm` returns its `TinyGPT` (which itself is the tokenizer-holder via `model.tokenizer`).
- **All chat** goes through `core/model_loader.chat()` → records metrics.
- **Embedding** uses `provider.embed()` with mean-pooling fallback for HF models.
- **MCP** uses JSON-RPC over HTTP at `/mcp` — protocol `2025-06-18`, 5 tools, 6 resources, 3 prompts.
- **Templates** are server-rendered Jinja2 + Chart.js (from CDN) — no build step.
