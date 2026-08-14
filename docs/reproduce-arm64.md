
# Reproducing the native Arm64 results

## Frozen comparison contract

Baseline and MCP Prism use the same Qwen2.5-1.5B-Instruct Q4_K_M model, llama.cpp server, user requests, system message, sampling settings, output limit, four CPU threads and native Arm64 host. Baseline receives all 61 tool schemas. Prism receives only the dependency-ready schema selected by the gateway.

Performance and quality are deliberately separate:

- performance generates the first token only and measures prompt tokens, TTFT, p50/p95 latency and requests/second;
- quality generates complete schema-constrained arguments and executes deterministic mock tool results through the entire workflow graph.

The performance schedule alternates Baseline and Prism order. It uses two warmups and seven recorded repetitions per workload with `cache_prompt` enabled. Workloads remain separate:

1. A — long common MCP prefix;
2. B — mixed realistic tool bundles;
3. C — low-sharing adversarial requests.

## Native host and pinned software

The final GitHub Actions host reported `uname -m = aarch64`, four Arm Neoverse-N2 cores and Ubuntu 24.04 Arm. The workflow pins llama.cpp `b9623`, builds with `GGML_CPU_KLEIDIAI=ON`, downloads the official Qwen GGUF from a revision-pinned URL, and records SHA-256 values.

## Commands

Start llama.cpp:

```bash
third_party/llama.cpp/build/bin/llama-server \
  -m models/qwen2.5-1.5b-instruct-q4_k_m.gguf \
  --host 127.0.0.1 --port 8080 -c 16384 -t 4 --jinja
```

Start MCP Prism:

```bash
PYTHONPATH=src python -m mcp_prism.cli proxy \
  --host 127.0.0.1 --port 8090 \
  --upstream http://127.0.0.1:8080
```

Run frozen held-out quality and cache-ON performance:

```bash
python scripts/evaluate_workflows.py \
  --data data/private/heldout_workflows_v3.json --held-out \
  --out results/arm64-quality/heldout-workflows-v3.json

python scripts/benchmark_proxy.py \
  --out results/arm64/raw.json --repetitions 7 --warmup 2 --cache-only
```

## Public evidence

- Quality: https://github.com/lsh2546/mcp-prism/actions/runs/31782960486
- Performance: https://github.com/lsh2546/mcp-prism/actions/runs/31783556529
- Machine-readable final manifest: `evidence/final-native-arm64.json`

The x64 files under `results/x64/` are development diagnostics and are never presented as Arm performance.

