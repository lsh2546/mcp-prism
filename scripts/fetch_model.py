from __future__ import annotations

import hashlib
import json
import urllib.request
from pathlib import Path


REVISION = "03b24027b56fc13a895539f88b0eeb8393803fa4"
BASE = f"https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/{REVISION}"
FILES = {
    "encoder-qint8-arm64.onnx": f"{BASE}/onnx/model_qint8_arm64.onnx?download=true",
    "tokenizer.json": f"{BASE}/tokenizer.json?download=true",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    destination = Path(__file__).resolve().parents[1] / "models"
    destination.mkdir(parents=True, exist_ok=True)
    manifest = {"source_revision": REVISION, "files": {}}
    for name, url in FILES.items():
        target = destination / name
        if not target.exists():
            print(f"Downloading {name}...")
            urllib.request.urlretrieve(url, target)
        manifest["files"][name] = {"sha256": sha256(target), "bytes": target.stat().st_size}
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
