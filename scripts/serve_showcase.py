#!/usr/bin/env python3
"""Serve the static showcase UI on http://127.0.0.1:8765/."""

from __future__ import annotations

import argparse
import functools
import http.server
import os
import socketserver
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SHOWCASE = ROOT / "showcase"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()

    if not (SHOWCASE / "index.html").exists():
        raise SystemExit("showcase/index.html missing. Build the UI first.")

    os.chdir(SHOWCASE)
    handler = functools.partial(http.server.SimpleHTTPRequestHandler)
    with socketserver.TCPServer((args.host, args.port), handler) as httpd:
        print(f"SGO-Audit showcase → http://{args.host}:{args.port}/")
        print("Ctrl+C to stop.")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nStopped.")


if __name__ == "__main__":
    main()
