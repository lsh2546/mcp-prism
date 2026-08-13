from __future__ import annotations

import argparse

from .proxy import serve


def main() -> int:
    parser = argparse.ArgumentParser(prog="mcp-prism")
    commands = parser.add_subparsers(dest="command", required=True)
    proxy = commands.add_parser("proxy")
    proxy.add_argument("--host", default="127.0.0.1")
    proxy.add_argument("--port", type=int, default=8090)
    proxy.add_argument("--upstream", default="http://127.0.0.1:8080")
    args = parser.parse_args()
    serve(args.host, args.port, args.upstream)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
