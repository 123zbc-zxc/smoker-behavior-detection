from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
FLAT_GROUP = ""


@dataclass(frozen=True)
class DatasetPair:
    group: str
    stem: str
    image_path: Path
    label_path: Path


def ensure_yolo_root(source_root: str | Path) -> Path:
    root = Path(source_root)
    images_root = root / "images"
    labels_root = root / "labels"
    if not images_root.exists() or not labels_root.exists():
        raise FileNotFoundError(
            f"Expected YOLO dataset root with `images/` and `labels/`: {root}"
        )
    return root


def discover_yolo_groups(source_root: str | Path) -> dict[str, tuple[Path, Path]]:
    root = ensure_yolo_root(source_root)
    images_root = root / "images"
    labels_root = root / "labels"

    label_subdirs = sorted(path for path in labels_root.iterdir() if path.is_dir())
    label_files = list(labels_root.glob("*.txt"))
    if label_subdirs and label_files:
        raise ValueError(
            f"Mixed label layout detected in {labels_root}; use either flat files or split subdirectories."
        )

    if label_subdirs:
        groups: dict[str, tuple[Path, Path]] = {}
        for label_dir in label_subdirs:
            image_dir = images_root / label_dir.name
            if not image_dir.exists():
                raise FileNotFoundError(
                    f"Missing image directory for split `{label_dir.name}`: {image_dir}"
                )
            groups[label_dir.name] = (image_dir, label_dir)
        return groups

    return {FLAT_GROUP: (images_root, labels_root)}


def build_image_index(image_dir: Path) -> dict[str, Path]:
    index: dict[str, Path] = {}
    for image_path in sorted(image_dir.iterdir()):
        if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        if image_path.stem in index:
            raise ValueError(f"Duplicate image stem `{image_path.stem}` found in {image_dir}")
        index[image_path.stem] = image_path
    return index


def collect_yolo_pairs(source_root: str | Path) -> tuple[list[DatasetPair], dict[str, list[dict[str, str]]]]:
    pairs: list[DatasetPair] = []
    issues = {"missing_images": [], "missing_labels": []}

    for group, (image_dir, label_dir) in discover_yolo_groups(source_root).items():
        image_index = build_image_index(image_dir)
        label_paths = sorted(label_dir.glob("*.txt"))
        label_stems = {path.stem for path in label_paths}

        for label_path in label_paths:
            image_path = image_index.get(label_path.stem)
            if image_path is None:
                issues["missing_images"].append(
                    {"group": group or "flat", "stem": label_path.stem, "label_path": str(label_path)}
                )
                continue
            pairs.append(
                DatasetPair(
                    group=group,
                    stem=label_path.stem,
                    image_path=image_path,
                    label_path=label_path,
                )
            )

        for stem, image_path in image_index.items():
            if stem in label_stems:
                continue
            issues["missing_labels"].append(
                {"group": group or "flat", "stem": stem, "image_path": str(image_path)}
            )

    pairs.sort(key=lambda item: ((item.group or "flat"), item.stem))
    return pairs, issues


def ensure_clean_yolo_output(output_root: str | Path, groups: Iterable[str]) -> Path:
    root = Path(output_root)
    normalized_groups = list(groups) or [FLAT_GROUP]
    for group in normalized_groups:
        image_dir = group_output_dir(root / "images", group)
        label_dir = group_output_dir(root / "labels", group)
        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)
        for directory in (image_dir, label_dir):
            for item in directory.iterdir():
                if item.is_file():
                    item.unlink()
    return root


def group_output_dir(base_dir: Path, group: str) -> Path:
    return base_dir if group == FLAT_GROUP else base_dir / group


def transfer_file(src: Path, dst: Path, mode: str = "copy") -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    if mode == "hardlink":
        try:
            os.link(src, dst)
            return
        except OSError:
            pass
    shutil.copy2(src, dst)
