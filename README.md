# MCP Prism

MCP Prism is an independent Arm64 Cloud AI project. It is an OpenAI-compatible optimization gateway that retrieves only the MCP tools relevant to each request, canonicalizes their schemas into stable shared prefixes, and drives llama.cpp prompt/KV reuse.

The project is not connected to, copied from, or dependent on AgentPulse or ArmForge.

## Current implementation status

Implemented source components:

- synchronous MCP stdio initialization and paginated `tools/list` collection;
- pinned INT8 ONNX semantic encoder that fails closed when its model is absent;
- multi-field tool representation and vector index;
- adaptive Top-K expansion and no-tool path;
- deterministic schema ordering and bundle fingerprints;
- OpenAI request transformation with `cache_prompt` and prefix metadata;
- exact quality metrics and non-regression gate;
- research, architecture, and measurement contracts.

Not yet complete:

- HTTP proxy server and llama.cpp connection;
- 50+ tool evaluation catalog;
- public/private benchmark ingestion;
- live baseline-versus-Prism dashboard;
- native aarch64 benchmark execution and optimization loop.

No performance claim is made before those items are complete and the native Arm64 gates pass.

## Retrieval model

Run `python scripts/fetch_model.py` to download the revision-pinned Apache-2.0 `model_qint8_arm64.onnx` and tokenizer from `sentence-transformers/all-MiniLM-L6-v2`. A manifest records SHA-256 digests. There is no lexical fallback in the product path.

## Quality contract

MCP Prism must not regress against the all-tools baseline on Recall@K, final tool accuracy, argument accuracy, task success, wrong-tool calls, or no-tool refusal. See `docs/architecture.md` and `docs/measurement-contract.md`.

## License

Apache-2.0.
