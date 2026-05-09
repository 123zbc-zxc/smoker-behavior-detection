from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (
    QApplication,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from scripts.yolo_utils import build_model


class SmokerDetectionWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Smoker Behavior Detection Demo")
        self.resize(1280, 800)

        self.model = None
        self.video_capture: cv2.VideoCapture | None = None
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_video_frame)

        self.status_label = QLabel("Model: not loaded")
        self.image_label = QLabel("Load a model, then choose an image or video.")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(960, 640)
        self.image_label.setStyleSheet("border: 1px solid #888; background: #111; color: #eee;")
        self.results_list = QListWidget()

        load_model_button = QPushButton("Load Weights")
        load_model_button.clicked.connect(self.load_model)
        load_image_button = QPushButton("Detect Image")
        load_image_button.clicked.connect(self.detect_image)
        load_video_button = QPushButton("Detect Video")
        load_video_button.clicked.connect(self.detect_video)
        stop_video_button = QPushButton("Stop Video")
        stop_video_button.clicked.connect(self.stop_video)

        button_layout = QHBoxLayout()
        for button in (load_model_button, load_image_button, load_video_button, stop_video_button):
            button_layout.addWidget(button)

        left_layout = QVBoxLayout()
        left_layout.addWidget(self.status_label)
        left_layout.addLayout(button_layout)
        left_layout.addWidget(self.image_label, stretch=1)

        right_layout = QVBoxLayout()
        right_layout.addWidget(QLabel("Detections"))
        right_layout.addWidget(self.results_list)

        root_layout = QHBoxLayout()
        root_layout.addLayout(left_layout, stretch=4)
        root_layout.addLayout(right_layout, stretch=1)

        container = QWidget()
        container.setLayout(root_layout)
        self.setCentralWidget(container)

    def load_model(self) -> None:
        weights_path, _ = QFileDialog.getOpenFileName(self, "Select model weights", str(ROOT), "PyTorch Weights (*.pt)")
        if not weights_path:
            return
        try:
            self.model = build_model(weights_path)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Model load failed", str(exc))
            return
        self.status_label.setText(f"Model: {weights_path}")

    def detect_image(self) -> None:
        if self.model is None:
            QMessageBox.warning(self, "No model", "Load model weights before running detection.")
            return
        image_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select image",
            str(ROOT),
            "Images (*.png *.jpg *.jpeg *.bmp)",
        )
        if not image_path:
            return

        results = self.model.predict(source=image_path, imgsz=416, conf=0.25, iou=0.45, device="cpu", verbose=False)
        if not results:
            return
        self.show_result(results[0])

    def detect_video(self) -> None:
        if self.model is None:
            QMessageBox.warning(self, "No model", "Load model weights before running detection.")
            return
        video_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select video",
            str(ROOT),
            "Videos (*.mp4 *.avi *.mov *.mkv)",
        )
        if not video_path:
            return

        self.stop_video()
        self.video_capture = cv2.VideoCapture(video_path)
        if not self.video_capture.isOpened():
            QMessageBox.critical(self, "Video error", f"Failed to open video: {video_path}")
            self.video_capture = None
            return
        self.timer.start(30)

    def update_video_frame(self) -> None:
        if self.video_capture is None or self.model is None:
            return

        ok, frame = self.video_capture.read()
        if not ok:
            self.stop_video()
            return

        results = self.model.predict(source=frame, imgsz=416, conf=0.25, iou=0.45, device="cpu", verbose=False)
        if results:
            self.show_result(results[0])

    def stop_video(self) -> None:
        self.timer.stop()
        if self.video_capture is not None:
            self.video_capture.release()
            self.video_capture = None

    def show_result(self, result) -> None:
        plotted = result.plot()
        rgb = cv2.cvtColor(plotted, cv2.COLOR_BGR2RGB)
        height, width, channel = rgb.shape
        image = QImage(rgb.data, width, height, channel * width, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(image).scaled(
            self.image_label.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )
        self.image_label.setPixmap(pixmap)

        self.results_list.clear()
        boxes = getattr(result, "boxes", None)
        names = getattr(result, "names", {})
        if boxes is None or boxes.cls is None:
            return

        for cls_id, conf in zip(boxes.cls.tolist(), boxes.conf.tolist()):
            class_name = names.get(int(cls_id), str(cls_id))
            QListWidgetItem(f"{class_name}: {conf:.3f}", self.results_list)


def main() -> None:
    app = QApplication(sys.argv)
    window = SmokerDetectionWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
