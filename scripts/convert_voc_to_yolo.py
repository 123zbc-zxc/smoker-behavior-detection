from __future__ import annotations

import json
import shutil
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path


CLASS_MAP = {"smoke": 2}


def voc_box_to_yolo(size: tuple[int, int], box: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    width, height = size
    xmin, ymin, xmax, ymax = box
    x_center = ((xmin + xmax) / 2.0) / width
    y_center = ((ymin + ymax) / 2.0) / height
    box_width = (xmax - xmin) / width
    box_height = (ymax - ymin) / height
    return x_center, y_center, box_width, box_height


def main() -> None:
    source_root = Path("datasets/raw/aistudio/pp_smoke")
    image_dir = source_root / "images"
    xml_dir = source_root / "Annotations"
    output_root = Path("datasets/interim/aistudio_yolo")
    output_image_dir = output_root / "images"
    output_label_dir = output_root / "labels"

    output_image_dir.mkdir(parents=True, exist_ok=True)
    output_label_dir.mkdir(parents=True, exist_ok=True)

    xml_files = sorted(xml_dir.glob("*.xml"))
    image_files = {p.stem: p for p in image_dir.iterdir() if p.is_file()}

    stats = {
        "source_root": str(source_root),
        "output_root": str(output_root),
        "image_count": 0,
        "xml_count": len(xml_files),
        "converted_files": 0,
        "skipped_missing_image": [],
        "skipped_unknown_class": [],
        "invalid_boxes": 0,
        "class_counts": Counter(),
    }

    for xml_path in xml_files:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        stem = xml_path.stem
        image_path = image_files.get(stem)
        if image_path is None:
            stats["skipped_missing_image"].append(stem)
            continue

        width = int(root.findtext("size/width", "0"))
        height = int(root.findtext("size/height", "0"))
        if width <= 0 or height <= 0:
            stats["invalid_boxes"] += 1
            continue

        yolo_lines: list[str] = []
        unknown_classes: list[str] = []
        for obj in root.findall("object"):
            name = obj.findtext("name", "").strip()
            if name not in CLASS_MAP:
                unknown_classes.append(name)
                continue

            box = obj.find("bndbox")
            if box is None:
                stats["invalid_boxes"] += 1
                continue

            xmin = float(box.findtext("xmin", "0"))
            ymin = float(box.findtext("ymin", "0"))
            xmax = float(box.findtext("xmax", "0"))
            ymax = float(box.findtext("ymax", "0"))

            xmin = max(0.0, min(xmin, width))
            xmax = max(0.0, min(xmax, width))
            ymin = max(0.0, min(ymin, height))
            ymax = max(0.0, min(ymax, height))

            if xmax <= xmin or ymax <= ymin:
                stats["invalid_boxes"] += 1
                continue

            x_center, y_center, box_width, box_height = voc_box_to_yolo(
                (width, height), (xmin, ymin, xmax, ymax)
            )
            class_id = CLASS_MAP[name]
            yolo_lines.append(
                f"{class_id} {x_center:.6f} {y_center:.6f} {box_width:.6f} {box_height:.6f}"
            )
            stats["class_counts"][name] += 1

        if unknown_classes:
            stats["skipped_unknown_class"].append(
                {"file": stem, "classes": sorted(set(unknown_classes))}
            )

        shutil.copy2(image_path, output_image_dir / image_path.name)
        (output_label_dir / f"{stem}.txt").write_text("\n".join(yolo_lines), encoding="utf-8")
        stats["converted_files"] += 1

    stats["image_count"] = len(image_files)
    stats["class_counts"] = dict(stats["class_counts"])

    report_path = Path("datasets/reports/aistudio_voc_to_yolo_report.json")
    report_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(stats, ensure_ascii=False, indent=2))
    print(f"\nReport saved to: {report_path}")


if __name__ == "__main__":
    main()
