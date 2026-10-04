# MCP — Model Context Protocol integration

SysMeAIEngine speaks MCP so desktop AI assistants, MCP-aware IDEs, and other clients
can call into it.

## Endpoint

```
POST http://127.0.0.1:11534/mcp
Content-Type: application/json

{"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}
```

## Methods

| Method | Result |
|---|---|
| `initialize` | `{protocolVersion, capabilities, serverInfo}` |
| `notifications/initialized` | `{}` |
| `ping` | `{}` |
| `tools/list` | list of tool descriptors |
| `tools/call` | run a tool `{name, arguments}` |
| `resources/list` | list of resource URIs |
| `resources/read` | content of a resource `{uri}` |
| `prompts/list` | prompt templates |
| `prompts/get` | build a prompt message |

## Tools

| Tool | Arguments | Description |
|---|---|---|
| `engine.chat` | `{prompt, model?, max_tokens?, temperature?}` | chat completion |
| `engine.embeddings` | `{input, model?}` | embed text(s) |
| `engine.models` | `{}` | list registered models |
| `engine.metrics` | `{}` | metrics snapshot |
| `engine.load` | `{name}` | load model |
| `rag.search` | `{query, corpus?, top_k?, method?}` | search local RAG corpora for relevant passages |
| `rag.list_corpora` | `{}` | list corpora with a built search index |

## Resources

| URI | Mime | Description |
|---|---|---|
| `engine://models` | json | full registry |
| `engine://loaded` | json | currently loaded model |
| `engine://metrics` | json | metrics snapshot |
| `engine://config` | json | non-secret engine config |
| `engine://agent/state` | json | self-learning agent history |
| `engine://datasets` | json | datasets in `data/datasets/` |

## Prompts

| Name | Arguments | Notes |
|---|---|---|
| `engine.chat` | prompt | one-shot chat |
| `engine.summarize` | text | 128-token summary |
| `engine.embed` | text | returns embedding dim |

## Sample

```bat
curl -X POST http://127.0.0.1:11534/mcp ^
  -H "Content-Type: application/json" ^
  -d "{\"jsonrpc\":\"2.0\",\"id\":1,\"method\":\"tools/call\",\"params\":{\"name\":\"engine.chat\",\"arguments\":{\"prompt\":\"hello\"}}}"
```
