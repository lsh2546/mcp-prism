
# MCP Prism ??Devpost submission copy

## Track

**Cloud AI**

## Tagline

Stop Arm64 AI agents from pre-filling every MCP tool schema on every request.

## Elevator pitch

MCP Prism is an OpenAI-compatible Arm64 inference gateway that retrieves only the next dependency-ready MCP tool, canonicalizes stable schema bundles, and reuses llama.cpp prompt/KV prefixes. On native Arm Neoverse-N2 it reaches up to **8.80횞 throughput**, improves frozen held-out workflow success from **60% to 85%**, and achieves **100% first-step accuracy**.

## Inspiration

An agent with dozens of MCP integrations often sends every tool name, description, and JSON Schema to the LLM?봢ven when a request needs only one tool. On CPU inference this turns irrelevant schemas into repeated prefill work. Existing prefix caches help only after a prefix is already identical. We wanted to reshape the request before inference so fewer tokens are processed and more prefixes become reusable.

## What it does

MCP Prism sits between an OpenAI-compatible agent and llama.cpp. It collects MCP tool definitions, embeds rich tool cards with a pinned INT8 ONNX encoder, decomposes requests into atomic tasks, identifies domain and service, combines semantic, lexical, action, schema, and dependency signals, and exposes only the next executable workflow step.

Selected schemas are canonicalized into stable order. Requests with the same tool bundle therefore share the same prefix fingerprint and can reuse llama.cpp's prompt/KV cache. If a request needs no tool, Prism sends none. For multi-step work, completed tool results unlock the next dependency-ready schema.

## How we built it

- Python OpenAI-compatible HTTP proxy
- MCP `tools/list` collection and pagination
- revision-pinned INT8 MiniLM ONNX retrieval model with verified SHA-256
- hierarchical service/action/schema routing and dependency-aware workflow planning
- deterministic JSON Schema normalization and prefix fingerprints
- llama.cpp `cache_prompt` integration
- Qwen2.5-1.5B-Instruct Q4_K_M on native Arm64
- llama.cpp b9623 built with KleidiAI
- public GitHub Actions benchmarks on four-core Arm Neoverse-N2 runners

Performance and quality are measured separately. Performance generates only the first token so decode does not hide prefill and TTFT effects. Quality runs complete schema-constrained arguments through deterministic mock tool results until each workflow finishes.

## Results

The final cache-ON native Arm64 run uses the same model, request, sampling settings, output limit, host, and four CPU threads for both paths. Baseline receives all 61 schemas; Prism receives one dependency-ready schema.

- **Peak throughput: 8.80횞** on the mixed realistic workload
- **All workloads: 5.60횞 to 8.80횞 throughput**
- **Mixed prompt tokens: 4,903 ??207**
- **Mixed p95 TTFT: 717 ms ??100 ms**
- **Held-out complete workflows: 60% ??85%** across 20 frozen requests
- **Development first-step accuracy: 100%**
- **Wrong first calls: 0%**

The low-sharing adversarial workload still reaches 7.49횞 throughput, so the headline is not produced only by repeated synthetic prompts.

## Challenges

Semantic similarity alone was not sufficient. It confused tools that share a domain but differ in service, operation, required inputs, or workflow position. We separated retrieval candidate recall from executable planning, added explicit service and action signals, and prevented dependent mutations from appearing before their source data exists.

We also found that small models can ignore named `tool_choice`. The gateway therefore owns the selected operation while the LLM generates schema-constrained arguments. This makes routing deterministic without replacing semantic retrieval with keyword search.

## Accomplishments

The project is a functioning proxy rather than an offline retrieval notebook. Existing OpenAI-compatible agents can adopt it by changing their base URL. It includes 61 overlapping MCP tools, multi-step and no-tool evaluation, raw per-request evidence, pinned model hashes, alternating benchmark order, three separately reported workloads, and automated native Arm64 runs.

## What we learned

The biggest optimization can happen before the model executes. Removing irrelevant structured context reduces prefill, increases throughput, lowers KV pressure, and can improve tool decisions by eliminating distracting choices. Stable canonicalization also turns semantically similar traffic into cacheable infrastructure.

## What's next

The same gateway design can support live MCP server discovery, larger catalogs, policy-aware routing, additional small rerankers, and production observability. The reusable core is already exposed as an OpenAI-compatible proxy with reproducible Arm64 evidence.

## Links

- Source: https://github.com/lsh2546/mcp-prism
- Native Arm64 performance: https://github.com/lsh2546/mcp-prism/actions/runs/31783556529
- Frozen held-out quality: https://github.com/lsh2546/mcp-prism/actions/runs/31782960486
- License: https://github.com/lsh2546/mcp-prism/blob/main/LICENSE

