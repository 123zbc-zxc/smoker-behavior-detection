from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.db import session_scope
from app.db_models import AlertEvent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Clear demo alert events while keeping alert rules.")
    parser.add_argument("--yes", action="store_true", help="Confirm deletion of all alert events.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.yes:
        raise SystemExit("Refusing to delete alert events without --yes.")
    with session_scope() as session:
        count = session.query(AlertEvent).count()
        session.query(AlertEvent).delete(synchronize_session=False)
    print(f"Deleted alert events: {count}")


if __name__ == "__main__":
    main()
