from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "figures"
ARCH_PATH = FIG_DIR / "图13_系统总体架构图.png"
DB_PATH = FIG_DIR / "图14_数据库关系说明图.png"


def load_font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        ("C:/Windows/Fonts/msyhbd.ttc" if bold else "C:/Windows/Fonts/msyh.ttc"),
        ("C:/Windows/Fonts/simhei.ttf" if bold else "C:/Windows/Fonts/simsun.ttc"),
    ]
    for path in candidates:
        font_path = Path(path)
        if font_path.exists():
            return ImageFont.truetype(str(font_path), size=size)
    return ImageFont.load_default()


TITLE_FONT = load_font(34, bold=True)
BOX_TITLE_FONT = load_font(24, bold=True)
TEXT_FONT = load_font(20)
SMALL_FONT = load_font(18)


def text_size(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> tuple[int, int]:
    left, top, right, bottom = draw.multiline_textbbox((0, 0), text, font=font, spacing=6)
    return right - left, bottom - top


def draw_centered_text(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    text: str,
    *,
    font: ImageFont.ImageFont,
    fill: str = "#1F2937",
) -> None:
    width, height = text_size(draw, text, font)
    x0, y0, x1, y1 = box
    x = x0 + (x1 - x0 - width) / 2
    y = y0 + (y1 - y0 - height) / 2
    draw.multiline_text((x, y), text, font=font, fill=fill, align="center", spacing=6)


def rounded_box(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    *,
    fill: str,
    outline: str,
    width: int = 3,
    radius: int = 22,
) -> None:
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=width)


def arrow(
    draw: ImageDraw.ImageDraw,
    start: tuple[int, int],
    end: tuple[int, int],
    *,
    fill: str = "#475569",
    width: int = 4,
) -> None:
    draw.line([start, end], fill=fill, width=width)
    ex, ey = end
    sx, sy = start
    if abs(ex - sx) >= abs(ey - sy):
        direction = 1 if ex >= sx else -1
        draw.polygon(
            [
                (ex, ey),
                (ex - 16 * direction, ey - 8),
                (ex - 16 * direction, ey + 8),
            ],
            fill=fill,
        )
    else:
        direction = 1 if ey >= sy else -1
        draw.polygon(
            [
                (ex, ey),
                (ex - 8, ey - 16 * direction),
                (ex + 8, ey - 16 * direction),
            ],
            fill=fill,
        )


def label(draw: ImageDraw.ImageDraw, position: tuple[int, int], text: str) -> None:
    draw.text(position, text, font=SMALL_FONT, fill="#334155")


def generate_architecture() -> None:
    image = Image.new("RGB", (1800, 1100), "#FAFAF9")
    draw = ImageDraw.Draw(image)

    draw.text((60, 36), "系统总体架构图", font=TITLE_FONT, fill="#0F172A")

    boxes = {
        "browser": (90, 180, 420, 320),
        "fastapi": (520, 150, 930, 350),
        "service": (1010, 150, 1420, 350),
        "weights": (1460, 110, 1730, 240),
        "reports": (1460, 260, 1730, 390),
        "database": (620, 500, 980, 690),
        "storage": (1030, 500, 1440, 690),
        "dataset": (120, 500, 520, 690),
        "training": (620, 820, 980, 980),
        "outputs": (1030, 820, 1440, 980),
    }

    rounded_box(draw, boxes["browser"], fill="#E0F2FE", outline="#0284C7")
    draw_centered_text(draw, boxes["browser"], "浏览器端页面\n首页 / 模型信息 / 图片检测 / 视频任务", font=BOX_TITLE_FONT)

    rounded_box(draw, boxes["fastapi"], fill="#DCFCE7", outline="#16A34A")
    draw_centered_text(
        draw,
        boxes["fastapi"],
        "FastAPI 接口层\napp/web_demo.py\n路由分发、模板渲染、参数校验",
        font=BOX_TITLE_FONT,
    )

    rounded_box(draw, boxes["service"], fill="#FDE68A", outline="#D97706")
    draw_centered_text(
        draw,
        boxes["service"],
        "推理服务层\nDetectionService\n模型加载、图像推理、视频处理",
        font=BOX_TITLE_FONT,
    )

    rounded_box(draw, boxes["weights"], fill="#EDE9FE", outline="#7C3AED")
    draw_centered_text(draw, boxes["weights"], "模型权重\nbest.pt / ONNX", font=BOX_TITLE_FONT)

    rounded_box(draw, boxes["reports"], fill="#FCE7F3", outline="#DB2777")
    draw_centered_text(draw, boxes["reports"], "实验摘要与报告\nruns/val / runs/reports", font=BOX_TITLE_FONT)

    rounded_box(draw, boxes["database"], fill="#F3E8FF", outline="#9333EA")
    draw_centered_text(
        draw,
        boxes["database"],
        "数据库层\nSQLite / PostgreSQL\n模型配置、检测记录、视频任务",
        font=BOX_TITLE_FONT,
    )

    rounded_box(draw, boxes["storage"], fill="#FEE2E2", outline="#DC2626")
    draw_centered_text(
        draw,
        boxes["storage"],
        "文件存储层\noutput/web_demo\n上传文件、结果图片、结果视频",
        font=BOX_TITLE_FONT,
    )

    rounded_box(draw, boxes["dataset"], fill="#E2E8F0", outline="#475569")
    draw_centered_text(
        draw,
        boxes["dataset"],
        "数据准备层\ndatasets/raw -> interim -> final\n标签映射、清洗、划分、检查",
        font=BOX_TITLE_FONT,
    )

    rounded_box(draw, boxes["training"], fill="#D1FAE5", outline="#059669")
    draw_centered_text(
        draw,
        boxes["training"],
        "训练与验证层\nscripts/train.py / val.py\n实验训练、验证、导出",
        font=BOX_TITLE_FONT,
    )

    rounded_box(draw, boxes["outputs"], fill="#FFEDD5", outline="#EA580C")
    draw_centered_text(
        draw,
        boxes["outputs"],
        "应用输出层\n检测框、历史记录、任务进度\n前端结果回显",
        font=BOX_TITLE_FONT,
    )

    arrow(draw, (420, 250), (520, 250))
    arrow(draw, (930, 250), (1010, 250))
    arrow(draw, (1420, 200), (1460, 175))
    arrow(draw, (1420, 300), (1460, 325))
    arrow(draw, (760, 350), (800, 500))
    arrow(draw, (1160, 350), (1200, 500))
    arrow(draw, (1440, 595), (1030, 900))
    arrow(draw, (980, 900), (1030, 900))
    arrow(draw, (520, 595), (620, 900))
    arrow(draw, (980, 900), (1460, 175))

    label(draw, (445, 224), "HTTP 请求")
    label(draw, (945, 224), "调用推理")
    label(draw, (1475, 144), "加载权重")
    label(draw, (1475, 293), "读取实验摘要")
    label(draw, (818, 420), "记录任务与状态")
    label(draw, (1212, 420), "保存结果文件")
    label(draw, (1070, 866), "结果回显")
    label(draw, (660, 860), "训练产物")
    label(draw, (1042, 152), "模型文件")

    note = (
        "说明：系统以浏览器端页面为展示入口，FastAPI 负责接口组织与页面渲染，"
        "DetectionService 负责实际推理；数据库保存结构化记录，文件系统保存大体积结果文件。"
    )
    draw.multiline_text((70, 1030), note, font=SMALL_FONT, fill="#475569", spacing=6)

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    image.save(ARCH_PATH)


def generate_database_diagram() -> None:
    image = Image.new("RGB", (1800, 980), "#FCFCFD")
    draw = ImageDraw.Draw(image)
    draw.text((60, 36), "数据库关系说明图", font=TITLE_FONT, fill="#0F172A")

    tables = {
        "model_registry": (120, 180, 520, 430),
        "app_settings": (660, 110, 1120, 320),
        "image_detections": (660, 390, 1120, 700),
        "image_detection_boxes": (1260, 390, 1710, 700),
        "video_tasks": (120, 530, 520, 840),
    }

    colors = {
        "model_registry": ("#DBEAFE", "#2563EB"),
        "app_settings": ("#DCFCE7", "#16A34A"),
        "image_detections": ("#FEF3C7", "#D97706"),
        "image_detection_boxes": ("#FCE7F3", "#DB2777"),
        "video_tasks": ("#EDE9FE", "#7C3AED"),
    }

    content = {
        "model_registry": "model_registry\nPK id\nname\nweights_path (unique)\ndevice\nis_available\nnote\ncreated_at / updated_at",
        "app_settings": "app_settings\nPK id\nFK default_model_id -> model_registry.id\ndefault_conf / default_iou\ndefault_imgsz\nmax_upload_mb",
        "image_detections": "image_detections\nPK id\nFK model_id -> model_registry.id\nsource_name / status\nmodel_name / weights_path\nconf / iou\nsource_image_path\nannotated_image_path\nnum_detections / created_at",
        "image_detection_boxes": "image_detection_boxes\nPK id\nFK detection_id -> image_detections.id\nclass_id / class_name\nconfidence\nx1 / y1 / x2 / y2",
        "video_tasks": "video_tasks\nPK id\nFK model_id -> model_registry.id\ntask_uuid (unique)\nstatus / progress\nprocessed_frames / total_frames\noutput_video_path\nsummary_json / error_message",
    }

    for key, box in tables.items():
        fill, outline = colors[key]
        rounded_box(draw, box, fill=fill, outline=outline, radius=18)
        x0, y0, x1, y1 = box
        draw.rectangle((x0, y0, x1, y0 + 48), fill=outline)
        draw_centered_text(draw, (x0, y0, x1, y0 + 48), key, font=BOX_TITLE_FONT, fill="#FFFFFF")
        draw.multiline_text((x0 + 18, y0 + 66), content[key].split("\n", 1)[1], font=TEXT_FONT, fill="#1F2937", spacing=6)

    arrow(draw, (520, 255), (660, 190))
    arrow(draw, (520, 640), (660, 540))
    arrow(draw, (1120, 545), (1260, 545))

    label(draw, (550, 195), "1 : 1 / 默认模型")
    label(draw, (548, 560), "1 : N / 检测记录")
    label(draw, (1160, 510), "1 : N / 框级结果")

    draw.line([(320, 430), (320, 530)], fill="#475569", width=4)
    draw.polygon([(320, 530), (312, 514), (328, 514)], fill="#475569")
    label(draw, (338, 468), "1 : N / 视频任务")

    note = (
        "说明：`model_registry` 统一保存可用权重与模型信息，`app_settings` 指定系统默认模型与阈值参数；"
        "`image_detections` 记录图片任务主表，`image_detection_boxes` 保存框级明细，`video_tasks` 记录视频任务状态与输出路径。"
    )
    draw.multiline_text((70, 905), note, font=SMALL_FONT, fill="#475569", spacing=6)

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    image.save(DB_PATH)


def main() -> None:
    generate_architecture()
    generate_database_diagram()
    print(f"Saved: {ARCH_PATH}")
    print(f"Saved: {DB_PATH}")


if __name__ == "__main__":
    main()
