"""Local desktop entry point; owns and cleans up its loopback Streamlit server."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import socket
import subprocess
import sys
import time
import webbrowser
from pathlib import Path
from urllib.request import ProxyHandler, build_opener

ROOT = Path(__file__).resolve().parent
NAME = "StyleMate" if (ROOT / "stylemate").is_dir() else "ResearchRadar"
APP = "app.py" if NAME == "StyleMate" else "web/app.py"


def server_command(port: int) -> list[str]:
    return [sys.executable, "-m", "streamlit", "run", str(ROOT / APP),
            "--server.address=127.0.0.1", f"--server.port={port}",
            "--server.headless=true", "--browser.gatherUsageStats=false",
            "--server.fileWatcherType=none"]


def available_port() -> int:
    with socket.socket() as connection:
        connection.bind(("127.0.0.1", 0))
        return connection.getsockname()[1]


def wait_ready(process, url: str, timeout: float = 45) -> None:
    deadline = time.monotonic() + timeout
    # Local health checks must not pass through a user-configured internet proxy.
    opener = build_opener(ProxyHandler({}))
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError("Local server exited; see data/desktop/server.log")
        try:
            with opener.open(url + "/_stcore/health", timeout=1) as response:
                if response.status == 200:
                    return
        except OSError:
            time.sleep(0.15)
    raise RuntimeError("Local server startup timed out; see data/desktop/server.log")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=f"{NAME} local desktop launcher")
    parser.add_argument("--browser", action="store_true", help="Use the default browser instead of a WebView window")
    parser.add_argument("--check", action="store_true", help="Report runtime readiness without starting anything")
    parser.add_argument("--no-ui", action="store_true", help="Start only the local server")
    parser.add_argument("--shutdown-after", type=float, default=0, help="Optional smoke-test lifetime in seconds")
    parser.add_argument("--port", type=int, default=0)
    args = parser.parse_args(argv)
    if not 0 <= args.port <= 65535 or args.shutdown_after < 0:
        parser.error("invalid port or shutdown time")
    ready = importlib.util.find_spec("streamlit") is not None
    native = importlib.util.find_spec("webview") is not None
    if args.check:
        print(json.dumps({"application": NAME, "streamlit": ready, "native_window": native,
                          "entry_exists": (ROOT / APP).is_file(), "host": "127.0.0.1"}))
        return 0 if ready and (ROOT / APP).is_file() else 1
    if not ready:
        print("Missing Streamlit. Install requirements.txt into this Python runtime.", file=sys.stderr)
        return 1
    log_dir = ROOT / "data" / "desktop"
    log_dir.mkdir(parents=True, exist_ok=True)
    port = args.port or available_port()
    url = f"http://127.0.0.1:{port}"
    environment = dict(os.environ, PYTHONUTF8="1", PYTHONNOUSERSITE="1")
    if NAME == "StyleMate":
        environment["STYLEMATE_DESKTOP"] = "1"
    process = None
    try:
        with (log_dir / "server.log").open("a", encoding="utf-8") as log:
            process = subprocess.Popen(server_command(port), cwd=ROOT, env=environment,
                                       stdout=log, stderr=log,
                                       creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            wait_ready(process, url)
            print(f"{NAME} ready at {url}", flush=True)
            if args.shutdown_after:
                time.sleep(args.shutdown_after)
            elif args.no_ui:
                process.wait()
            elif native and not args.browser:
                import webview
                webview.create_window(NAME, url, width=1240, height=860, min_size=(900, 620))
                webview.start(private_mode=True, storage_path=str(log_dir / "webview"))
            else:
                webbrowser.open(url)
                print("Close this launcher with Ctrl+C to stop the local server.", flush=True)
                process.wait()
        return 0
    except KeyboardInterrupt:
        return 0
    except (OSError, RuntimeError) as error:
        print(str(error), file=sys.stderr)
        return 1
    finally:
        # Only stop the exact child process created by this invocation.
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=8)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)


if __name__ == "__main__":
    raise SystemExit(main())
