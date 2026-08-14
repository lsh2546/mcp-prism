
# MCP Prism

> An OpenAI-compatible gateway that stops Arm64 AI agents from pre-filling every MCP tool schema on every request.

## Native Arm64 results

| Result | MCP Prism | Baseline | Evidence |
|---|---:|---:|---|
| Peak cache-ON throughput | **12.50 req/s** | 1.42 req/s | **8.80횞** |
| Held-out complete workflow success | **85%** | 60% | +25 percentage points |
| Development first-step accuracy | **100%** | 33.3% | wrong first call: **0%** |
| Prompt tokens, mixed workload | **207** | 4,903 | 95.8% fewer |

Measured on a native `aarch64` GitHub runner with a 4-core Arm Neoverse-N2 CPU, Qwen2.5-1.5B-Instruct Q4_K_M, llama.cpp `b9623`, KleidiAI, identical requests and generation settings. Performance uses first-token generation; full-output quality is evaluated separately.

- [Final native Arm64 performance run](https://github.com/lsh2546/mcp-prism/actions/runs/31783556529)
- [Frozen held-out workflow run](https://github.com/lsh2546/mcp-prism/actions/runs/31782960486)
- [Evidence manifest](evidence/final-native-arm64.json)
- [Live comparison dashboard](dashboard/index.html)

## What it does

Most MCP agents serialize every connected tool definition into every LLM request. MCP Prism collects those definitions once, retrieves the relevant tools with a pinned INT8 ONNX semantic model, canonicalizes their JSON Schemas, and exposes only the next dependency-ready operation. Stable bundles create reusable llama.cpp prompt/KV prefixes.

The gateway supports:

- MCP `tools/list` collection with pagination;
- INT8 semantic retrieval plus domain, service, action, schema and workflow signals;
- deterministic JSON Schema normalization and stable tool ordering;
- adaptive candidate expansion and a no-tool path;
- dependency-aware multi-step workflows;
- OpenAI-compatible `/v1/chat/completions` proxying;
- llama.cpp `cache_prompt` reuse;
- raw per-request quality and performance evidence.

## Quick start

```bash
python -m pip install --only-binary=:all: -e .
python scripts/fetch_model.py
python -m mcp_prism.cli proxy --host 127.0.0.1 --port 8090 --upstream http://127.0.0.1:8080
```

Point an OpenAI-compatible agent at `http://127.0.0.1:8090/v1`. Use header `x-mcp-prism-mode: prism`; use `baseline` to send all 61 schemas.

## Reproduce the Arm64 evidence

The benchmark and quality workflows are manual so published evidence cannot be changed by an unrelated push.

```bash
python scripts/evaluate_workflows.py \
  --data data/private/heldout_workflows_v3.json \
  --held-out --out results/arm64-quality/heldout-workflows-v3.json

python scripts/benchmark_proxy.py \
  --out results/arm64/raw.json --repetitions 7 --warmup 2 --cache-only
```

See [the full reproduction contract](docs/reproduce-arm64.md) for the pinned model, hardware metadata, comparison rules, workloads and gate calculations.

## Repository map

- `src/mcp_prism/` ??gateway, retrieval, canonicalization and proxy product code
- `data/private/tool_catalog.json` ??61 realistic, overlapping MCP tools
- `data/private/heldout_workflows_v3.json` ??frozen 20-request held-out workflow set
- `scripts/` ??model fetch, quality evaluation and alternating benchmark
- `.github/workflows/` ??native Arm64 quality and performance automation
- `evidence/` ??final metric manifest and public run links
- `dashboard/` ??judge-facing baseline-versus-Prism comparison

## License

Apache-2.0. The repository root contains the full [LICENSE](LICENSE). The retrieval model is revision-pinned and Apache-2.0; model hashes and upstream provenance are recorded in `models/manifest.json`.

MCP Prism is an independent project and does not reuse AgentPulse or ArmForge code.

