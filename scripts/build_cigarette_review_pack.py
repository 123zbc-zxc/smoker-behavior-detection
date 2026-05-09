from __future__ import annotations

import argparse
import csv
import html
import json
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.yolo_utils import dump_json, ensure_exists, load_yaml


CLASS_NAMES = {
    0: "cigarette",
    1: "smoking_person",
    2: "smoke",
}

BOX_COLORS = {
    0: "#d9480f",
    1: "#1d6fd6",
    2: "#2f9e44",
}


@dataclass
class ReviewSample:
    image_name: str
    split: str
    image_path: Path
    label_path: Path
    preview_path: Path
    copied_image_path: Path
    copied_label_path: Path
    image_size: tuple[int, int]
    image_area: int
    class_counts: dict[str, int]
    cigarette_count: int
    cigarette_area_ratio_min: float | None
    cigarette_width_px_min: float | None
    cigarette_height_px_min: float | None
    cigarette_aspect_ratio_max: float | None
    suspicious_small_box: bool
    suspicious_reason: str
    multi_cigarette: bool
    low_resolution_image: bool
    low_resolution_reason: str
    slender_cigarette: bool
    slender_reason: str
    review_group: str = "other_priority"
    small_box_rank: int | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a human-review pack for cigarette hard cases from a priority manifest."
    )
    parser.add_argument(
        "--manifest",
        default="runs/reports/cigarette_priority_review.txt",
        help="TXT file listing image file names that should be reviewed.",
    )
    parser.add_argument(
        "--data",
        default="configs/data_smoking_balanced.yaml",
        help="Dataset YAML used to resolve image and label roots.",
    )
    parser.add_argument(
        "--output-dir",
        default="tmp/cigarette_priority_review",
        help="Output folder for copied samples, previews, and review index files.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional max number of samples to pack. 0 means all samples in the manifest.",
    )
    parser.add_argument(
        "--tiny-area-threshold",
        type=float,
        default=0.0005,
        help="If any cigarette box area ratio is below this threshold, flag the sample as suspicious.",
    )
    parser.add_argument(
        "--small-side-px-threshold",
        type=float,
        default=12.0,
        help="If any cigarette box width or height in pixels is below this threshold, flag the sample as suspicious.",
    )
    parser.add_argument(
        "--rank-smallest-count",
        type=int,
        default=10,
        help="Always rank the smallest cigarette-box samples by min area ratio, even if no hard threshold is hit.",
    )
    parser.add_argument(
        "--multi-cigarette-threshold",
        type=int,
        default=2,
        help="Flag images with at least this many cigarette boxes as multi-cigarette hard cases.",
    )
    parser.add_argument(
        "--low-res-min-side-threshold",
        type=int,
        default=384,
        help="Flag images with a shorter side below this value as low-resolution hard cases.",
    )
    parser.add_argument(
        "--low-res-area-threshold",
        type=int,
        default=200000,
        help="Flag images with width*height below this value as low-resolution hard cases.",
    )
    parser.add_argument(
        "--slender-aspect-threshold",
        type=float,
        default=2.5,
        help="Flag cigarette boxes whose long-side/short-side ratio exceeds this value.",
    )
    return parser.parse_args()


def resolve_dataset_roots(data_path: Path) -> tuple[Path, dict[str, Path], dict[str, Path]]:
    config = load_yaml(data_path)
    dataset_root = Path(config["path"])
    if not dataset_root.is_absolute():
        dataset_root = (ROOT / dataset_root).resolve()
    image_roots = {
        split: dataset_root / config[split]
        for split in ("train", "val", "test")
    }
    label_roots = {
        split: dataset_root / "labels" / split
        for split in ("train", "val", "test")
    }
    return dataset_root, image_roots, label_roots


def load_manifest(path: Path, limit: int) -> list[str]:
    names = [
        line.strip()
        for line in path.read_text(encoding="utf-8", errors="ignore").splitlines()
        if line.strip()
    ]
    deduped = list(dict.fromkeys(names))
    if limit > 0:
        return deduped[:limit]
    return deduped


def index_dataset(image_roots: dict[str, Path]) -> dict[str, tuple[str, Path]]:
    index: dict[str, tuple[str, Path]] = {}
    for split, root in image_roots.items():
        for image_path in sorted(root.iterdir()):
            if image_path.is_file():
                index[image_path.name] = (split, image_path)
    return index


def parse_label_file(label_path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not label_path.exists():
        return rows
    for line_no, line in enumerate(
        label_path.read_text(encoding="utf-8", errors="ignore").splitlines(),
        start=1,
    ):
        parts = line.split()
        if len(parts) != 5:
            continue
        class_id, x, y, w, h = parts
        try:
            rows.append(
                {
                    "line_no": line_no,
                    "class_id": int(class_id),
                    "x": float(x),
                    "y": float(y),
                    "w": float(w),
                    "h": float(h),
                }
            )
        except ValueError:
            continue
    return rows


def yolo_to_xyxy(box: dict[str, Any], width: int, height: int) -> tuple[float, float, float, float]:
    cx = float(box["x"]) * width
    cy = float(box["y"]) * height
    bw = float(box["w"]) * width
    bh = float(box["h"]) * height
    return (
        cx - bw / 2,
        cy - bh / 2,
        cx + bw / 2,
        cy + bh / 2,
    )


def build_preview(
    image_path: Path,
    label_path: Path,
    preview_path: Path,
    *,
    tiny_area_threshold: float,
    small_side_px_threshold: float,
    multi_cigarette_threshold: int,
    low_res_min_side_threshold: int,
    low_res_area_threshold: int,
    slender_aspect_threshold: float,
) -> tuple[tuple[int, int], dict[str, int], dict[str, Any]]:
    preview_path.parent.mkdir(parents=True, exist_ok=True)
    class_counter: Counter[str] = Counter()
    cigarette_area_ratios: list[float] = []
    cigarette_widths_px: list[float] = []
    cigarette_heights_px: list[float] = []
    cigarette_aspect_ratios: list[float] = []
    with Image.open(image_path) as img:
        canvas = img.convert("RGB")
        draw = ImageDraw.Draw(canvas)
        width, height = canvas.size
        for box in parse_label_file(label_path):
            class_id = int(box["class_id"])
            class_name = CLASS_NAMES.get(class_id, str(class_id))
            class_counter[class_name] += 1
            x1, y1, x2, y2 = yolo_to_xyxy(box, width, height)
            color = BOX_COLORS.get(class_id, "#111111")
            if class_id == 0:
                cigarette_area_ratios.append(float(box["w"]) * float(box["h"]))
                width_px = abs(x2 - x1)
                height_px = abs(y2 - y1)
                cigarette_widths_px.append(width_px)
                cigarette_heights_px.append(height_px)
                short_side = max(min(width_px, height_px), 1.0)
                long_side = max(width_px, height_px)
                cigarette_aspect_ratios.append(long_side / short_side)
            draw.rectangle((x1, y1, x2, y2), outline=color, width=3)
            label = f"{class_name}"
            draw.rectangle((x1, max(0, y1 - 18), x1 + 110, y1), fill=color)
            draw.text((x1 + 4, max(1, y1 - 16)), label, fill="white")
        canvas.save(preview_path)
        stats = {
            "cigarette_area_ratio_min": min(cigarette_area_ratios) if cigarette_area_ratios else None,
            "cigarette_width_px_min": min(cigarette_widths_px) if cigarette_widths_px else None,
            "cigarette_height_px_min": min(cigarette_heights_px) if cigarette_heights_px else None,
            "cigarette_aspect_ratio_max": max(cigarette_aspect_ratios) if cigarette_aspect_ratios else None,
        }
        suspicious_reasons: list[str] = []
        if stats["cigarette_area_ratio_min"] is not None and stats["cigarette_area_ratio_min"] < tiny_area_threshold:
            suspicious_reasons.append(
                f"min area ratio {stats['cigarette_area_ratio_min']:.6f} < {tiny_area_threshold:.6f}"
            )
        if stats["cigarette_width_px_min"] is not None and stats["cigarette_width_px_min"] < small_side_px_threshold:
            suspicious_reasons.append(
                f"min width {stats['cigarette_width_px_min']:.1f}px < {small_side_px_threshold:.1f}px"
            )
        if stats["cigarette_height_px_min"] is not None and stats["cigarette_height_px_min"] < small_side_px_threshold:
            suspicious_reasons.append(
                f"min height {stats['cigarette_height_px_min']:.1f}px < {small_side_px_threshold:.1f}px"
            )
        stats["suspicious_small_box"] = bool(suspicious_reasons)
        stats["suspicious_reason"] = "; ".join(suspicious_reasons)
        stats["multi_cigarette"] = class_counter.get("cigarette", 0) >= max(multi_cigarette_threshold, 2)
        low_res_reasons: list[str] = []
        shorter_side = min(width, height)
        image_area = width * height
        if shorter_side < low_res_min_side_threshold:
            low_res_reasons.append(f"short side {shorter_side}px < {low_res_min_side_threshold}px")
        if image_area < low_res_area_threshold:
            low_res_reasons.append(f"image area {image_area} < {low_res_area_threshold}")
        stats["low_resolution_image"] = bool(low_res_reasons)
        stats["low_resolution_reason"] = "; ".join(low_res_reasons)
        slender_reasons: list[str] = []
        if (
            stats["cigarette_aspect_ratio_max"] is not None
            and stats["cigarette_aspect_ratio_max"] >= slender_aspect_threshold
        ):
            slender_reasons.append(
                f"max aspect ratio {stats['cigarette_aspect_ratio_max']:.2f} >= {slender_aspect_threshold:.2f}"
            )
        stats["slender_cigarette"] = bool(slender_reasons)
        stats["slender_reason"] = "; ".join(slender_reasons)
        return (width, height), dict(class_counter), stats


def build_review_pack(
    manifest_names: list[str],
    image_index: dict[str, tuple[str, Path]],
    label_roots: dict[str, Path],
    output_dir: Path,
    *,
    tiny_area_threshold: float,
    small_side_px_threshold: float,
    multi_cigarette_threshold: int,
    low_res_min_side_threshold: int,
    low_res_area_threshold: int,
    slender_aspect_threshold: float,
) -> tuple[list[ReviewSample], list[str]]:
    samples: list[ReviewSample] = []
    missing: list[str] = []

    images_dir = output_dir / "images"
    labels_dir = output_dir / "labels"
    previews_dir = output_dir / "previews"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    previews_dir.mkdir(parents=True, exist_ok=True)

    for image_name in manifest_names:
        if image_name not in image_index:
            missing.append(image_name)
            continue

        split, image_path = image_index[image_name]
        label_path = label_roots[split] / f"{image_path.stem}.txt"
        copied_image_path = images_dir / image_name
        copied_label_path = labels_dir / f"{image_path.stem}.txt"
        preview_path = previews_dir / image_name

        copied_image_path.write_bytes(image_path.read_bytes())
        if label_path.exists():
            copied_label_path.write_bytes(label_path.read_bytes())
        else:
            copied_label_path.write_text("", encoding="utf-8")

        image_size, class_counts, stats = build_preview(
            image_path,
            label_path,
            preview_path,
            tiny_area_threshold=tiny_area_threshold,
            small_side_px_threshold=small_side_px_threshold,
            multi_cigarette_threshold=multi_cigarette_threshold,
            low_res_min_side_threshold=low_res_min_side_threshold,
            low_res_area_threshold=low_res_area_threshold,
            slender_aspect_threshold=slender_aspect_threshold,
        )
        samples.append(
            ReviewSample(
                image_name=image_name,
                split=split,
                image_path=image_path,
                label_path=label_path,
                preview_path=preview_path,
                copied_image_path=copied_image_path,
                copied_label_path=copied_label_path,
                image_size=image_size,
                image_area=image_size[0] * image_size[1],
                class_counts=class_counts,
                cigarette_count=class_counts.get("cigarette", 0),
                cigarette_area_ratio_min=stats["cigarette_area_ratio_min"],
                cigarette_width_px_min=stats["cigarette_width_px_min"],
                cigarette_height_px_min=stats["cigarette_height_px_min"],
                cigarette_aspect_ratio_max=stats["cigarette_aspect_ratio_max"],
                suspicious_small_box=stats["suspicious_small_box"],
                suspicious_reason=stats["suspicious_reason"],
                multi_cigarette=stats["multi_cigarette"],
                low_resolution_image=stats["low_resolution_image"],
                low_resolution_reason=stats["low_resolution_reason"],
                slender_cigarette=stats["slender_cigarette"],
                slender_reason=stats["slender_reason"],
            )
        )

    return samples, missing


def write_summary_csv(path: Path, samples: list[ReviewSample]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "image_name",
                "split",
                "width",
                "height",
                "image_area",
                "cigarette_count",
                "cigarette_area_ratio_min",
                "cigarette_width_px_min",
                "cigarette_height_px_min",
                "cigarette_aspect_ratio_max",
                "suspicious_small_box",
                "suspicious_reason",
                "multi_cigarette",
                "low_resolution_image",
                "low_resolution_reason",
                "slender_cigarette",
                "slender_reason",
                "review_group",
                "small_box_rank",
                "class_counts_json",
                "preview_path",
                "image_path",
                "label_path",
            ]
        )
        for sample in samples:
            writer.writerow(
                [
                    sample.image_name,
                    sample.split,
                    sample.image_size[0],
                    sample.image_size[1],
                    sample.image_area,
                    sample.cigarette_count,
                    sample.cigarette_area_ratio_min,
                    sample.cigarette_width_px_min,
                    sample.cigarette_height_px_min,
                    sample.cigarette_aspect_ratio_max,
                    sample.suspicious_small_box,
                    sample.suspicious_reason,
                    sample.multi_cigarette,
                    sample.low_resolution_image,
                    sample.low_resolution_reason,
                    sample.slender_cigarette,
                    sample.slender_reason,
                    sample.review_group,
                    sample.small_box_rank,
                    json.dumps(sample.class_counts, ensure_ascii=False),
                    str(sample.preview_path),
                    str(sample.image_path),
                    str(sample.label_path),
                ]
            )


def write_index_html(
    path: Path,
    missing: list[str],
    suspicious_samples: list[ReviewSample],
    ranked_smallest_samples: list[ReviewSample],
    multi_cigarette_samples: list[ReviewSample],
    low_resolution_samples: list[ReviewSample],
    slender_samples: list[ReviewSample],
    regular_samples: list[ReviewSample],
) -> None:
    total_count = len({sample.image_name for sample in [*suspicious_samples, *ranked_smallest_samples, *regular_samples]})

    def format_number(value: float | None, digits: int = 6, suffix: str = "") -> str:
        if value is None:
            return "-"
        return f"{value:.{digits}f}{suffix}"

    def render_cards(items: list[ReviewSample]) -> str:
        if not items:
            return '<p class="empty">No samples in this group.</p>'
        cards = []
        for sample in items:
            badge = ""
            if sample.suspicious_small_box:
                badge = f'<p><strong>small-box flag:</strong> {html.escape(sample.suspicious_reason)}</p>'
            elif sample.review_group == "ranked_smallest_candidate" and sample.small_box_rank is not None:
                badge = f"<p><strong>ranked candidate:</strong> top {sample.small_box_rank} smallest cigarette box in this review pack</p>"
            extra_tags: list[str] = []
            if sample.multi_cigarette:
                extra_tags.append("multi-cigarette")
            if sample.low_resolution_image:
                extra_tags.append("low-resolution")
            if sample.slender_cigarette:
                extra_tags.append("slender-box")
            tag_block = ""
            if extra_tags:
                tag_block = f"<p><strong>tags:</strong> {html.escape(', '.join(extra_tags))}</p>"
            extra_reason_lines = []
            if sample.low_resolution_reason:
                extra_reason_lines.append(
                    f"<p><strong>low-res flag:</strong> {html.escape(sample.low_resolution_reason)}</p>"
                )
            if sample.slender_reason:
                extra_reason_lines.append(
                    f"<p><strong>slender flag:</strong> {html.escape(sample.slender_reason)}</p>"
                )
            card_class = "card"
            if sample.review_group in {"suspicious_small_box", "ranked_smallest_candidate"}:
                card_class += " priority"
            cards.append(
                f"""
            <article class="{card_class}">
              <img src="previews/{html.escape(sample.image_name)}" alt="{html.escape(sample.image_name)}">
              <div class="meta">
                <h3>{html.escape(sample.image_name)}</h3>
                <p><strong>split:</strong> {html.escape(sample.split)}</p>
                <p><strong>size:</strong> {sample.image_size[0]}x{sample.image_size[1]}</p>
                <p><strong>group:</strong> {html.escape(sample.review_group)}</p>
                <p><strong>min cigarette area ratio:</strong> {format_number(sample.cigarette_area_ratio_min)}</p>
                <p><strong>min cigarette side:</strong> {format_number(sample.cigarette_width_px_min, digits=1, suffix='px')} / {format_number(sample.cigarette_height_px_min, digits=1, suffix='px')}</p>
                <p><strong>max cigarette aspect ratio:</strong> {format_number(sample.cigarette_aspect_ratio_max, digits=2)}</p>
                <p><strong>class counts:</strong> {html.escape(json.dumps(sample.class_counts, ensure_ascii=False))}</p>
                {badge}
                {tag_block}
                {''.join(extra_reason_lines)}
                <p><a href="images/{html.escape(sample.image_name)}">open image</a> | <a href="labels/{html.escape(sample.copied_label_path.name)}">open label</a></p>
              </div>
            </article>
            """
            )
        return "".join(cards)

    missing_block = ""
    if missing:
        missing_block = (
            "<section><h2>Missing from dataset index</h2><pre>"
            + html.escape("\n".join(missing))
            + "</pre></section>"
        )

    page = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Cigarette Priority Review Pack</title>
  <style>
    body {{ font-family: Segoe UI, Arial, sans-serif; margin: 24px; background: #f7f4ee; color: #18242a; }}
    h1 {{ margin-bottom: 8px; }}
    h2 {{ margin-top: 28px; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 18px; }}
    .card {{ background: white; border: 1px solid #d8d2c7; border-radius: 16px; overflow: hidden; box-shadow: 0 8px 24px rgba(0,0,0,0.06); }}
    .priority {{ border: 2px solid #d9480f; }}
    .empty {{ padding: 12px 14px; background: white; border: 1px dashed #d8d2c7; border-radius: 12px; }}
    .card img {{ width: 100%; display: block; background: #eee6d8; }}
    .meta {{ padding: 14px; }}
    .meta h3 {{ margin: 0 0 8px; font-size: 16px; word-break: break-all; }}
    .meta p {{ margin: 6px 0; line-height: 1.5; }}
    pre {{ background: white; padding: 12px; border-radius: 12px; border: 1px solid #d8d2c7; white-space: pre-wrap; }}
  </style>
</head>
<body>
  <h1>Cigarette Priority Review Pack</h1>
  <p>Total samples: {total_count}</p>
  <p>Suspicious small-box samples: {len(suspicious_samples)}</p>
  <p>Ranked smallest-box candidates: {len(ranked_smallest_samples)}</p>
  <p>Multi-cigarette hard cases: {len(multi_cigarette_samples)}</p>
  <p>Low-resolution hard cases: {len(low_resolution_samples)}</p>
  <p>Slender-cigarette hard cases: {len(slender_samples)}</p>
  {missing_block}
  <section>
    <h2>Suspicious Small-Box First</h2>
    <div class="grid">
      {render_cards(suspicious_samples)}
    </div>
  </section>
  <section>
    <h2>Ranked Smallest Cigarette Boxes</h2>
    <div class="grid">
      {render_cards(ranked_smallest_samples)}
    </div>
  </section>
  <section>
    <h2>Multi-Cigarette Hard Cases</h2>
    <div class="grid">
      {render_cards(multi_cigarette_samples)}
    </div>
  </section>
  <section>
    <h2>Low-Resolution Hard Cases</h2>
    <div class="grid">
      {render_cards(low_resolution_samples)}
    </div>
  </section>
  <section>
    <h2>Slender Cigarette Hard Cases</h2>
    <div class="grid">
      {render_cards(slender_samples)}
    </div>
  </section>
  <section>
    <h2>Other Priority Samples</h2>
    <div class="grid">
      {render_cards(regular_samples)}
    </div>
  </section>
</body>
</html>
"""
    path.write_text(page, encoding="utf-8")


def assign_review_groups(samples: list[ReviewSample], rank_smallest_count: int) -> tuple[list[ReviewSample], list[ReviewSample], list[ReviewSample]]:
    suspicious_samples = [sample for sample in samples if sample.suspicious_small_box]

    ranked_candidates = sorted(
        [sample for sample in samples if sample.cigarette_area_ratio_min is not None and not sample.suspicious_small_box],
        key=lambda sample: (
            sample.cigarette_area_ratio_min,
            sample.cigarette_width_px_min if sample.cigarette_width_px_min is not None else float("inf"),
        ),
    )[: max(rank_smallest_count, 0)]
    ranked_by_name = {sample.image_name: index for index, sample in enumerate(ranked_candidates, start=1)}

    for sample in samples:
        sample.small_box_rank = None
        if sample.suspicious_small_box:
            sample.review_group = "suspicious_small_box"
            continue
        if sample.image_name in ranked_by_name:
            sample.review_group = "ranked_smallest_candidate"
            sample.small_box_rank = ranked_by_name[sample.image_name]
            continue
        sample.review_group = "other_priority"

    regular_samples = [sample for sample in samples if sample.review_group == "other_priority"]
    return suspicious_samples, ranked_candidates, regular_samples


def write_group_lists(output_dir: Path, groups: dict[str, list[ReviewSample]]) -> None:
    groups_dir = output_dir / "groups"
    groups_dir.mkdir(parents=True, exist_ok=True)
    for group_name, items in groups.items():
        (groups_dir / f"{group_name}.txt").write_text(
            "\n".join(sample.image_name for sample in items),
            encoding="utf-8",
        )


def main() -> None:
    args = parse_args()
    manifest_path = ensure_exists(args.manifest, "Priority review manifest")
    data_path = ensure_exists(args.data, "Dataset config")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    _, image_roots, label_roots = resolve_dataset_roots(data_path)
    manifest_names = load_manifest(manifest_path, args.limit)
    image_index = index_dataset(image_roots)
    samples, missing = build_review_pack(
        manifest_names,
        image_index,
        label_roots,
        output_dir,
        tiny_area_threshold=args.tiny_area_threshold,
        small_side_px_threshold=args.small_side_px_threshold,
        multi_cigarette_threshold=args.multi_cigarette_threshold,
        low_res_min_side_threshold=args.low_res_min_side_threshold,
        low_res_area_threshold=args.low_res_area_threshold,
        slender_aspect_threshold=args.slender_aspect_threshold,
    )
    suspicious_samples, ranked_smallest_samples, regular_samples = assign_review_groups(
        samples,
        args.rank_smallest_count,
    )
    multi_cigarette_samples = [sample for sample in samples if sample.multi_cigarette]
    low_resolution_samples = [sample for sample in samples if sample.low_resolution_image]
    slender_samples = [sample for sample in samples if sample.slender_cigarette]

    summary = {
        "manifest": str(manifest_path),
        "data": str(data_path),
        "output_dir": str(output_dir),
        "requested_count": len(manifest_names),
        "packed_count": len(samples),
        "suspicious_small_box_count": len(suspicious_samples),
        "ranked_smallest_candidate_count": len(ranked_smallest_samples),
        "multi_cigarette_count": len(multi_cigarette_samples),
        "low_resolution_count": len(low_resolution_samples),
        "slender_cigarette_count": len(slender_samples),
        "other_priority_count": len(regular_samples),
        "missing_count": len(missing),
        "missing_images": missing,
        "suspicious_small_box_images": [sample.image_name for sample in suspicious_samples],
        "ranked_smallest_candidate_images": [sample.image_name for sample in ranked_smallest_samples],
        "multi_cigarette_images": [sample.image_name for sample in multi_cigarette_samples],
        "low_resolution_images": [sample.image_name for sample in low_resolution_samples],
        "slender_cigarette_images": [sample.image_name for sample in slender_samples],
        "other_priority_images": [sample.image_name for sample in regular_samples],
        "samples": [
            {
                "image_name": sample.image_name,
                "split": sample.split,
                "image_size": list(sample.image_size),
                "image_area": sample.image_area,
                "cigarette_count": sample.cigarette_count,
                "cigarette_area_ratio_min": sample.cigarette_area_ratio_min,
                "cigarette_width_px_min": sample.cigarette_width_px_min,
                "cigarette_height_px_min": sample.cigarette_height_px_min,
                "cigarette_aspect_ratio_max": sample.cigarette_aspect_ratio_max,
                "suspicious_small_box": sample.suspicious_small_box,
                "suspicious_reason": sample.suspicious_reason,
                "multi_cigarette": sample.multi_cigarette,
                "low_resolution_image": sample.low_resolution_image,
                "low_resolution_reason": sample.low_resolution_reason,
                "slender_cigarette": sample.slender_cigarette,
                "slender_reason": sample.slender_reason,
                "review_group": sample.review_group,
                "small_box_rank": sample.small_box_rank,
                "class_counts": sample.class_counts,
                "preview_path": str(sample.preview_path),
                "image_path": str(sample.image_path),
                "label_path": str(sample.label_path),
            }
            for sample in samples
        ],
    }

    dump_json(output_dir / "review_summary.json", summary)
    (output_dir / "suspicious_small_box.txt").write_text(
        "\n".join(sample.image_name for sample in suspicious_samples),
        encoding="utf-8",
    )
    (output_dir / "smallest_ranked_samples.txt").write_text(
        "\n".join(sample.image_name for sample in ranked_smallest_samples),
        encoding="utf-8",
    )
    write_group_lists(
        output_dir,
        {
            "suspicious_small_box": suspicious_samples,
            "ranked_smallest_candidate": ranked_smallest_samples,
            "multi_cigarette": multi_cigarette_samples,
            "low_resolution": low_resolution_samples,
            "slender_cigarette": slender_samples,
            "other_priority": regular_samples,
        },
    )
    write_summary_csv(output_dir / "review_summary.csv", samples)
    write_index_html(
        output_dir / "index.html",
        missing,
        suspicious_samples,
        ranked_smallest_samples,
        multi_cigarette_samples,
        low_resolution_samples,
        slender_samples,
        regular_samples,
    )
    print(f"Review pack saved to: {output_dir}")


if __name__ == "__main__":
    main()
