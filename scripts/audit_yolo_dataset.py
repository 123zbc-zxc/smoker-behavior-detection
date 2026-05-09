from __future__ import annotations

import argparse
import csv
import json
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
PIL_FORMATS_ALLOWED_BY_YOLO = {"JPEG", "PNG", "BMP", "WEBP", "TIFF"}
CLASS_NAMES = {0: "cigarette", 1: "smoking_person", 2: "smoke"}
SPLITS = ("train", "val", "test")


@dataclass(frozen=True)
class Box:
    cls: int
    x: float
    y: float
    w: float
    h: float
    line_no: int
    raw: str

    @property
    def area(self) -> float:
        return self.w * self.h

    @property
    def aspect(self) -> float:
        return max(self.w / max(self.h, 1e-9), self.h / max(self.w, 1e-9))

    @property
    def xyxy(self) -> tuple[float, float, float, float]:
        return (
            self.x - self.w / 2.0,
            self.y - self.h / 2.0,
            self.x + self.w / 2.0,
            self.y + self.h / 2.0,
        )


def parse_args() -> argparse.Namespace:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    parser = argparse.ArgumentParser(description="Deep audit a YOLO dataset and export suspicious-label reports.")
    parser.add_argument("--dataset-root", default="datasets/final/smoke_bal", help="Dataset root containing images/ and labels/.")
    parser.add_argument("--output-dir", default=f"datasets/reports/smoke_bal_audit_{stamp}", help="Report output directory.")
    parser.add_argument("--preview-limit", type=int, default=120, help="Max suspicious samples copied with preview images.")
    parser.add_argument(
        "--check-image-header",
        action="store_true",
        help="Open every image header to catch unsupported formats. Slower on large datasets.",
    )
    parser.add_argument("--tiny-area", type=float, default=0.00025, help="Boxes smaller than this normalized area are severe.")
    parser.add_argument("--small-cigarette-area", type=float, default=0.0008, help="Cigarette boxes below this area need review.")
    parser.add_argument("--small-person-area", type=float, default=0.008, help="smoking_person boxes below this area need review.")
    parser.add_argument("--small-smoke-area", type=float, default=0.003, help="Smoke boxes below this area need review.")
    parser.add_argument("--huge-cigarette-area", type=float, default=0.08, help="Cigarette boxes above this area need review.")
    parser.add_argument("--huge-box-area", type=float, default=0.85, help="Any box above this area needs review.")
    parser.add_argument("--aspect-threshold", type=float, default=14.0, help="Very thin/wide boxes above this aspect ratio need review.")
    parser.add_argument("--duplicate-iou", type=float, default=0.95, help="Same-class boxes above this IoU are duplicate candidates.")
    return parser.parse_args()


def repo_path(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return Path(__file__).resolve().parents[1] / candidate


def image_files(path: Path) -> list[Path]:
    if not path.exists():
        return []
    return sorted(p for p in path.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def label_files(path: Path) -> list[Path]:
    if not path.exists():
        return []
    return sorted(p for p in path.iterdir() if p.is_file() and p.suffix.lower() == ".txt")


def parse_label_file(label_path: Path) -> tuple[list[Box], list[dict[str, Any]]]:
    issues: list[dict[str, Any]] = []
    boxes: list[Box] = []
    text = label_path.read_text(encoding="utf-8", errors="ignore")
    for line_no, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        parts = stripped.split()
        if len(parts) != 5:
            issues.append({"issue_type": "invalid_label_line", "line_no": line_no, "raw": stripped})
            continue
        try:
            cls_f, x, y, w, h = map(float, parts)
        except ValueError:
            issues.append({"issue_type": "invalid_label_value", "line_no": line_no, "raw": stripped})
            continue
        cls = int(cls_f)
        if cls_f != cls or cls not in CLASS_NAMES:
            issues.append({"issue_type": "unknown_class", "line_no": line_no, "class_id": parts[0], "raw": stripped})
            continue
        if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0 and 0.0 < w <= 1.0 and 0.0 < h <= 1.0):
            issues.append({"issue_type": "out_of_range_box", "line_no": line_no, "class_id": cls, "raw": stripped})
            continue
        boxes.append(Box(cls=cls, x=x, y=y, w=w, h=h, line_no=line_no, raw=stripped))
    return boxes, issues


def iou_norm(a: Box, b: Box) -> float:
    ax1, ay1, ax2, ay2 = a.xyxy
    bx1, by1, bx2, by2 = b.xyxy
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    union = a.area + b.area - inter
    return inter / union if union > 0 else 0.0


def add_issue(
    issues: list[dict[str, Any]],
    *,
    severity: str,
    issue_type: str,
    split: str,
    image_path: Path | None,
    label_path: Path | None,
    box: Box | None = None,
    message: str = "",
    extra: dict[str, Any] | None = None,
) -> None:
    row: dict[str, Any] = {
        "severity": severity,
        "issue_type": issue_type,
        "split": split,
        "class_id": "" if box is None else box.cls,
        "class_name": "" if box is None else CLASS_NAMES.get(box.cls, str(box.cls)),
        "image_path": "" if image_path is None else str(image_path),
        "label_path": "" if label_path is None else str(label_path),
        "line_no": "" if box is None else box.line_no,
        "x": "" if box is None else round(box.x, 6),
        "y": "" if box is None else round(box.y, 6),
        "w": "" if box is None else round(box.w, 6),
        "h": "" if box is None else round(box.h, 6),
        "area": "" if box is None else round(box.area, 8),
        "aspect": "" if box is None else round(box.aspect, 3),
        "message": message,
    }
    if extra:
        row.update(extra)
    issues.append(row)


def score_issue(row: dict[str, Any]) -> tuple[int, float]:
    severity_rank = {"error": 0, "warning": 1, "review": 2}.get(str(row.get("severity")), 3)
    area = row.get("area")
    try:
        area_value = float(area)
    except (TypeError, ValueError):
        area_value = 1.0
    return severity_rank, area_value


def audit_box(box: Box, args: argparse.Namespace) -> list[tuple[str, str, str]]:
    out: list[tuple[str, str, str]] = []
    if box.area < args.tiny_area:
        out.append(("warning", "tiny_box", "box is extremely small; verify whether the label is real"))
    if box.cls == 0 and box.area < args.small_cigarette_area:
        out.append(("review", "small_cigarette_review", "small cigarette box; useful but high risk, verify placement"))
    if box.cls == 0 and box.area > args.huge_cigarette_area:
        out.append(("warning", "huge_cigarette_box", "cigarette box is unusually large"))
    if box.cls == 1 and box.area < args.small_person_area:
        out.append(("warning", "small_smoking_person_box", "smoking_person box is unusually small"))
    if box.cls == 2 and box.area < args.small_smoke_area:
        out.append(("review", "small_smoke_region", "smoke region is tiny; may be ambiguous"))
    if box.area > args.huge_box_area:
        out.append(("warning", "huge_box", "box covers most of the image"))
    if box.aspect > args.aspect_threshold:
        out.append(("review", "extreme_aspect_ratio", "box is very thin or very wide"))
    x1, y1, x2, y2 = box.xyxy
    if x1 < -1e-6 or y1 < -1e-6 or x2 > 1.0 + 1e-6 or y2 > 1.0 + 1e-6:
        out.append(("warning", "box_crosses_image_edge", "normalized box extends beyond image edge"))
    return out


def draw_preview(image_path: Path, label_path: Path, boxes: list[Box], issue_rows: list[dict[str, Any]], output_path: Path) -> None:
    with Image.open(image_path) as img:
        img = img.convert("RGB")
        draw = ImageDraw.Draw(img)
        w, h = img.size
        flagged_lines = {int(r["line_no"]) for r in issue_rows if str(r.get("line_no", "")).isdigit()}
        colors = {0: (230, 57, 70), 1: (42, 111, 219), 2: (46, 160, 67)}
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy
            px = [x1 * w, y1 * h, x2 * w, y2 * h]
            color = colors.get(box.cls, (50, 50, 50))
            width = 4 if box.line_no in flagged_lines else 2
            draw.rectangle(px, outline=color, width=width)
            draw.text((px[0], max(0, px[1] - 14)), f"{box.line_no}:{CLASS_NAMES.get(box.cls)}", fill=color)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        img.save(output_path, quality=92)
    _ = label_path


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "severity",
        "issue_type",
        "split",
        "class_id",
        "class_name",
        "image_path",
        "label_path",
        "line_no",
        "x",
        "y",
        "w",
        "h",
        "area",
        "aspect",
        "message",
        "extra",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def write_markdown(path: Path, summary: dict[str, Any], issue_counts: Counter[str], top_rows: list[dict[str, Any]]) -> None:
    lines = [
        "# YOLO 数据集体检报告",
        "",
        f"- 数据集: `{summary['dataset_root']}`",
        f"- 图片总数: `{summary['total_images']}`",
        f"- 标签文件总数: `{summary['total_label_files']}`",
        f"- 标注框总数: `{summary['total_boxes']}`",
        f"- 可疑项总数: `{summary['total_issues']}`",
        "",
        "## 类别分布",
        "",
    ]
    for cls_id, name in CLASS_NAMES.items():
        lines.append(f"- `{cls_id} {name}`: `{summary['class_counts'].get(str(cls_id), 0)}`")
    lines.extend(["", "## 可疑项类型统计", ""])
    for key, value in issue_counts.most_common():
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(["", "## 优先检查样本", ""])
    for row in top_rows[:30]:
        lines.append(
            f"- `{row.get('severity')}` `{row.get('issue_type')}` "
            f"{row.get('split')} {row.get('class_name')} line {row.get('line_no')}: "
            f"`{row.get('image_path')}`"
        )
    lines.extend(
        [
            "",
            "## 处理建议",
            "",
            "- 先处理 `error` 和 `warning`，尤其是缺失文件、越界框、重复框、异常大/异常小框。",
            "- `small_cigarette_review` 不一定是错标；香烟本来就是小目标，但这类样本最影响召回率，适合优先人工抽查。",
            "- 不建议把全部可疑项都手工修完；答辩阶段优先修 80-150 张代表性 hard-case 即可。",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    dataset_root = repo_path(args.dataset_root)
    output_dir = repo_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_issues: list[dict[str, Any]] = []
    issues_by_sample: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    boxes_by_sample: dict[tuple[str, str], list[Box]] = {}
    image_by_sample: dict[tuple[str, str], Path] = {}
    label_by_sample: dict[tuple[str, str], Path] = {}
    class_counts: Counter[str] = Counter()
    split_counts: dict[str, Any] = {}
    total_boxes = 0

    for split in SPLITS:
        image_dir = dataset_root / "images" / split
        label_dir = dataset_root / "labels" / split
        images = image_files(image_dir)
        labels = label_files(label_dir)
        image_stems = {p.stem: p for p in images}
        label_stems = {p.stem: p for p in labels}
        split_info = {"images": len(images), "label_files": len(labels), "boxes": 0, "class_counts": Counter()}

        for stem, image_path in sorted(image_stems.items()):
            if args.check_image_header:
                try:
                    with Image.open(image_path) as img:
                        if img.format and img.format.upper() not in PIL_FORMATS_ALLOWED_BY_YOLO:
                            add_issue(
                                all_issues,
                                severity="error",
                                issue_type="unsupported_image_format",
                                split=split,
                                image_path=image_path,
                                label_path=label_dir / f"{stem}.txt",
                                message=f"image header format is {img.format}; YOLO training will ignore this file",
                            )
                except Exception as exc:  # noqa: BLE001
                    add_issue(
                        all_issues,
                        severity="error",
                        issue_type="corrupt_image",
                        split=split,
                        image_path=image_path,
                        label_path=label_dir / f"{stem}.txt",
                        message=f"image cannot be opened: {exc}",
                    )
            if stem not in label_stems:
                add_issue(
                    all_issues,
                    severity="error",
                    issue_type="missing_label_file",
                    split=split,
                    image_path=image_path,
                    label_path=label_dir / f"{stem}.txt",
                    message="image has no matching label file",
                )

        for stem, label_path in sorted(label_stems.items()):
            if stem not in image_stems:
                add_issue(
                    all_issues,
                    severity="error",
                    issue_type="missing_image_file",
                    split=split,
                    image_path=image_dir / f"{stem}.jpg",
                    label_path=label_path,
                    message="label file has no matching image",
                )
                continue

            image_path = image_stems[stem]
            text = label_path.read_text(encoding="utf-8", errors="ignore").strip()
            if not text:
                add_issue(
                    all_issues,
                    severity="warning",
                    issue_type="empty_label_file",
                    split=split,
                    image_path=image_path,
                    label_path=label_path,
                    message="empty label file; verify if this is a true negative",
                )

            boxes, parse_issues = parse_label_file(label_path)
            sample_key = (split, stem)
            boxes_by_sample[sample_key] = boxes
            image_by_sample[sample_key] = image_path
            label_by_sample[sample_key] = label_path
            for issue in parse_issues:
                add_issue(
                    all_issues,
                    severity="error",
                    issue_type=str(issue["issue_type"]),
                    split=split,
                    image_path=image_path,
                    label_path=label_path,
                    message=str(issue),
                    extra={"line_no": issue.get("line_no", "")},
                )

            line_keys: Counter[str] = Counter()
            for box in boxes:
                line_keys[f"{box.cls} {box.x:.6f} {box.y:.6f} {box.w:.6f} {box.h:.6f}"] += 1
                class_counts[str(box.cls)] += 1
                split_info["class_counts"][str(box.cls)] += 1
                split_info["boxes"] += 1
                total_boxes += 1
                for severity, issue_type, message in audit_box(box, args):
                    add_issue(
                        all_issues,
                        severity=severity,
                        issue_type=issue_type,
                        split=split,
                        image_path=image_path,
                        label_path=label_path,
                        box=box,
                        message=message,
                    )
            for duplicated_line, count in line_keys.items():
                if count > 1:
                    add_issue(
                        all_issues,
                        severity="warning",
                        issue_type="duplicate_label_line",
                        split=split,
                        image_path=image_path,
                        label_path=label_path,
                        message=f"same YOLO label appears {count} times: {duplicated_line}",
                    )
            for idx, a in enumerate(boxes):
                for b in boxes[idx + 1 :]:
                    if a.cls == b.cls and iou_norm(a, b) >= args.duplicate_iou:
                        add_issue(
                            all_issues,
                            severity="warning",
                            issue_type="overlapping_duplicate_box",
                            split=split,
                            image_path=image_path,
                            label_path=label_path,
                            box=a,
                            message=f"same-class boxes overlap heavily, line {a.line_no} and {b.line_no}",
                            extra={"extra": f"other_line={b.line_no};iou={iou_norm(a, b):.4f}"},
                        )
        split_info["class_counts"] = dict(split_info["class_counts"])
        split_counts[split] = split_info

    for row in all_issues:
        image_path = str(row.get("image_path", ""))
        if image_path:
            key = (str(row.get("split", "")), Path(image_path).stem)
            issues_by_sample[key].append(row)

    sorted_issues = sorted(all_issues, key=score_issue)
    issue_counts = Counter(str(row["issue_type"]) for row in all_issues)
    severity_counts = Counter(str(row["severity"]) for row in all_issues)
    summary = {
        "dataset_root": str(dataset_root),
        "output_dir": str(output_dir),
        "total_images": sum(v["images"] for v in split_counts.values()),
        "total_label_files": sum(v["label_files"] for v in split_counts.values()),
        "total_boxes": total_boxes,
        "total_issues": len(all_issues),
        "severity_counts": dict(severity_counts),
        "issue_counts": dict(issue_counts),
        "class_counts": dict(class_counts),
        "splits": split_counts,
        "thresholds": {
            "tiny_area": args.tiny_area,
            "small_cigarette_area": args.small_cigarette_area,
            "small_person_area": args.small_person_area,
            "small_smoke_area": args.small_smoke_area,
            "huge_cigarette_area": args.huge_cigarette_area,
            "huge_box_area": args.huge_box_area,
            "aspect_threshold": args.aspect_threshold,
            "duplicate_iou": args.duplicate_iou,
        },
    }

    (output_dir / "dataset_audit_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "suspicious_labels.json").write_text(json.dumps(sorted_issues, ensure_ascii=False, indent=2), encoding="utf-8")
    write_csv(output_dir / "suspicious_labels.csv", sorted_issues)
    write_markdown(output_dir / "dataset_audit_report.md", summary, issue_counts, sorted_issues)

    preview_dir = output_dir / "previews"
    (preview_dir / "images").mkdir(parents=True, exist_ok=True)
    (preview_dir / "labels").mkdir(parents=True, exist_ok=True)
    copied = 0
    seen_samples: set[tuple[str, str]] = set()
    for row in sorted_issues:
        key = (str(row.get("split", "")), Path(str(row.get("image_path", ""))).stem)
        if key in seen_samples or key not in image_by_sample:
            continue
        seen_samples.add(key)
        image_path = image_by_sample[key]
        label_path = label_by_sample.get(key, Path(str(row.get("label_path", ""))))
        dst_base = f"{key[0]}_{key[1]}"
        shutil.copy2(image_path, preview_dir / "images" / image_path.name)
        if label_path.exists():
            shutil.copy2(label_path, preview_dir / "labels" / label_path.name)
        draw_preview(image_path, label_path, boxes_by_sample.get(key, []), issues_by_sample.get(key, []), preview_dir / f"{dst_base}.jpg")
        copied += 1
        if copied >= args.preview_limit:
            break

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Audit report saved to: {output_dir}")


if __name__ == "__main__":
    main()
