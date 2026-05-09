from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the FastAPI smoker behavior web demo.")
    parser.add_argument("--host", default="127.0.0.1", help="Host address for the demo server.")
    parser.add_argument("--port", type=int, default=8000, help="Port for the demo server.")
    parser.add_argument("--reload", action="store_true", help="Enable auto reload in development.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        import uvicorn
    except ImportError as exc:
        raise RuntimeError(
            "uvicorn is not installed. Run `pip install -r requirements.txt` first."
        ) from exc

    uvicorn.run("app.web_demo:app", host=args.host, port=args.port, reload=args.reload)


if __name__ == "__main__":
    main()
