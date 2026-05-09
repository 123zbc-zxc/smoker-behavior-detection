from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.inspect_checkpoint import summarize_checkpoint
from scripts.yolo_utils import ensure_exists, project_root, register_custom_modules, require_ultralytics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Resume YOLO training from an explicit checkpoint and optionally sync weights to a backup folder."
    )
    parser.add_argument("--checkpoint", required=True, help="Path to last.pt used for resume.")
    parser.add_argument("--device", default="0", help="Training device override, e.g. 0 or cpu.")
    parser.add_argument("--workers", type=int, default=2, help="Worker override for the resumed session.")
    parser.add_argument("--cache", default="disk", help="Cache mode override, e.g. disk, ram, or false.")
    parser.add_argument("--backup-dir", help="Optional folder that receives synced last.pt / best.pt backups.")
    parser.add_argument(
        "--sync-last-every",
        type=int,
        default=2,
        help="Sync last.pt to backup-dir every N epochs. Ignored if backup-dir is not set.",
    )
    parser.add_argument(
        "--snapshot-every",
        type=int,
        default=5,
        help="Write numbered last_e{epoch}.pt snapshots every N epochs. Ignored if backup-dir is not set.",
    )
    return parser.parse_args()


def normalize_cache(value: str) -> str | bool:
    lowered = value.strip().lower()
    if lowered in {"false", "0", "none", "off"}:
        return False
    if lowered in {"true", "1", "on"}:
        return True
    return value


def build_backup_callback(backup_dir: Path, sync_last_every: int, snapshot_every: int):
    backup_dir.mkdir(parents=True, exist_ok=True)

    def on_train_epoch_end(trainer) -> None:
        epoch = int(getattr(trainer, "epoch", -1)) + 1
        save_dir = Path(getattr(trainer, "save_dir"))
        weights_dir = save_dir / "weights"
        last_pt = weights_dir / "last.pt"
        best_pt = weights_dir / "best.pt"

        if last_pt.exists() and sync_last_every > 0 and epoch % sync_last_every == 0:
            shutil.copy2(last_pt, backup_dir / "last.pt")

        if last_pt.exists() and snapshot_every > 0 and epoch % snapshot_every == 0:
            snapshot_path = backup_dir / f"last_e{epoch}.pt"
            shutil.copy2(last_pt, snapshot_path)
            print(f"Synced snapshot to: {snapshot_path}")

        if best_pt.exists():
            shutil.copy2(best_pt, backup_dir / "best.pt")

    return on_train_epoch_end


def main() -> None:
    args = parse_args()
    checkpoint = ensure_exists(args.checkpoint, "Resume checkpoint")
    summary = summarize_checkpoint(checkpoint)
    print("Checkpoint summary:", summary)

    YOLO = require_ultralytics()
    register_custom_modules()
    model = YOLO(str(checkpoint))

    if args.backup_dir:
        backup_dir = Path(args.backup_dir)
        if not backup_dir.is_absolute():
            backup_dir = (project_root() / backup_dir).resolve()
        model.add_callback(
            "on_train_epoch_end",
            build_backup_callback(
                backup_dir=backup_dir,
                sync_last_every=max(args.sync_last_every, 0),
                snapshot_every=max(args.snapshot_every, 0),
            ),
        )
        print(f"Backup sync enabled: {backup_dir}")

    results = model.train(
        resume=True,
        device=args.device,
        workers=args.workers,
        cache=normalize_cache(args.cache),
    )
    print(f"Training artifacts saved to: {getattr(results, 'save_dir', '')}")


if __name__ == "__main__":
    main()
