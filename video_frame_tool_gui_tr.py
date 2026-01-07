# video_frame_tool_gui.py
# Requirements:
#   pip install pyqt5 opencv-python
# Run:
#   python video_frame_tool_gui.py

import os
import math
import sys
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any

import cv2
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QLabel,
    QPushButton,
    QFileDialog,
    QVBoxLayout,
    QHBoxLayout,
    QGroupBox,
    QFormLayout,
    QLineEdit,
    QSpinBox,
    QDoubleSpinBox,
    QRadioButton,
    QButtonGroup,
    QMessageBox,
    QFrame,
    QProgressBar,
    QSizePolicy,
)


# -------------------- Core helpers --------------------


def get_video_metadata(video_path: str) -> Dict[str, Any]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Video açılamadı!")

    fps = float(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration_sec = (total_frames / fps) if fps > 0 else 0.0

    cap.release()
    return {
        "fps": fps,
        "total_frames": total_frames,
        "duration_sec": duration_sec,
        "resolution": (width, height),
    }


def _clamp_int(x: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, x))


def secs_to_frame_range(
    start_sec: Optional[float], end_sec: Optional[float], fps: float, total_frames: int
) -> Tuple[Optional[int], Optional[int]]:
    # [start_sec, end_sec) mantığı (end hariç)
    if start_sec is None and end_sec is None:
        return None, None

    if start_sec is None:
        start_sec = 0.0
    if end_sec is None:
        end_sec = (total_frames / fps) if fps > 0 else 0.0

    if start_sec < 0 or end_sec < 0:
        raise ValueError("Saniye değerleri negatif olamaz.")
    if start_sec > end_sec:
        raise ValueError("start_sec end_sec'den büyük olamaz.")

    start_frame = int(math.floor(start_sec * fps))
    end_frame = int(math.floor(end_sec * fps)) - 1
    return start_frame, end_frame


def normalize_range(
    meta: Dict[str, Any],
    mode: str,
    start_frame: Optional[int] = None,
    end_frame: Optional[int] = None,
    start_sec: Optional[float] = None,
    end_sec: Optional[float] = None,
) -> Tuple[int, int]:
    fps = float(meta["fps"])
    total_frames = int(meta["total_frames"])

    if total_frames <= 0:
        raise ValueError("Video frame sayısı 0 görünüyor.")
    if fps <= 0:
        raise ValueError("Video FPS değeri 0/negatif görünüyor.")

    if mode == "seconds":
        start_frame, end_frame = secs_to_frame_range(
            start_sec, end_sec, fps, total_frames
        )

    # default full video
    s = 0 if start_frame is None else int(start_frame)
    e = (total_frames - 1) if end_frame is None else int(end_frame)

    s = _clamp_int(s, 0, total_frames - 1)
    e = _clamp_int(e, 0, total_frames - 1)

    if s > e:
        raise ValueError(f"Başlangıç ({s}) bitiş ({e}) değerinden büyük olamaz.")

    return s, e


def compute_frame_interval(source_fps: float, target_fps: Optional[float]) -> float:
    # target_fps None/0 => her kare
    if target_fps is None or float(target_fps) <= 0:
        return 1.0
    target_fps = float(target_fps)
    if target_fps >= source_fps:
        return 1.0
    return source_fps / target_fps


def read_specific_frame(video_path: str, frame_index: int):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Video açılamadı!")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        raise ValueError("Video frame sayısı okunamadı.")

    frame_index = min(max(0, int(frame_index)), total - 1)

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        raise ValueError("İstenen frame okunamadı.")

    return frame, frame_index


def cv_frame_to_qpixmap(frame_bgr) -> QPixmap:
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb.shape
    bytes_per_line = ch * w
    qimg = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
    return QPixmap.fromImage(qimg)


# -------------------- Resizable preview label --------------------


class PreviewLabel(QLabel):
    def __init__(self):
        super().__init__()
        self._original_pixmap: Optional[QPixmap] = None
        self.setAlignment(Qt.AlignCenter)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMinimumHeight(260)

    def set_preview_pixmap(self, pixmap: QPixmap):
        self._original_pixmap = pixmap
        self._rescale()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._rescale()

    def _rescale(self):
        if self._original_pixmap is None:
            return
        w = max(1, self.width() - 16)
        h = max(1, self.height() - 16)
        scaled = self._original_pixmap.scaled(
            w, h, Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self.setPixmap(scaled)


# -------------------- Worker thread for extraction --------------------


@dataclass
class ExtractParams:
    video_path: str
    output_dir: str
    start_frame: int
    end_frame: int
    target_fps: Optional[float]
    image_ext: str = "jpg"
    jpeg_quality: int = 95


class ExtractWorker(QThread):
    progress = pyqtSignal(int)
    finished_ok = pyqtSignal(int, str)  # saved_count, output_dir
    failed = pyqtSignal(str)

    def __init__(self, params: ExtractParams):
        super().__init__()
        self.params = params

    def run(self):
        try:
            p = self.params
            meta = get_video_metadata(p.video_path)
            source_fps = float(meta["fps"])
            interval = compute_frame_interval(source_fps, p.target_fps)

            os.makedirs(p.output_dir, exist_ok=True)
            cap = cv2.VideoCapture(p.video_path)
            if not cap.isOpened():
                raise ValueError("Video açılamadı!")

            cap.set(cv2.CAP_PROP_POS_FRAMES, p.start_frame)

            ext = p.image_ext.lower().lstrip(".")
            imwrite_params = []
            if ext in ("jpg", "jpeg"):
                imwrite_params = [int(cv2.IMWRITE_JPEG_QUALITY), int(p.jpeg_quality)]

            total_to_scan = p.end_frame - p.start_frame + 1
            saved = 0
            cur = p.start_frame
            next_capture = float(p.start_frame)
            last_emit = -1

            while cap.isOpened() and cur <= p.end_frame:
                ret, frame = cap.read()
                if not ret:
                    break

                if float(cur) + 1e-9 >= next_capture:
                    filename = f"frame_{cur:06d}_{saved:06d}.{ext}"
                    out_path = os.path.join(p.output_dir, filename)
                    cv2.imwrite(out_path, frame, imwrite_params)
                    saved += 1
                    next_capture += interval

                scanned = cur - p.start_frame + 1
                pct = int((scanned / max(1, total_to_scan)) * 100)
                if pct != last_emit:
                    self.progress.emit(pct)
                    last_emit = pct

                cur += 1

            cap.release()
            self.progress.emit(100)
            self.finished_ok.emit(saved, p.output_dir)

        except Exception as e:
            self.failed.emit(str(e))


# -------------------- GUI --------------------


class DropArea(QFrame):
    file_dropped = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.setAcceptDrops(True)
        self.setFrameStyle(QFrame.StyledPanel | QFrame.Raised)

        self.label = QLabel(
            "🎬 Videoyu buraya sürükleyip bırak\nveya 'Dosya Seç' ile aç", self
        )
        self.label.setAlignment(Qt.AlignCenter)

        lay = QVBoxLayout(self)
        lay.addWidget(self.label)

    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls():
            e.acceptProposedAction()

    def dropEvent(self, e):
        urls = e.mimeData().urls()
        if not urls:
            return
        path = urls[0].toLocalFile()
        if path:
            self.file_dropped.emit(path)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Video Frame Extractor (PyQt)")
        self.resize(1100, 650)

        self.video_path: Optional[str] = None
        self.meta: Optional[Dict[str, Any]] = None
        self.worker: Optional[ExtractWorker] = None

        # --- Left
        self.drop = DropArea()
        self.drop.file_dropped.connect(self.load_video)

        self.btn_pick = QPushButton("Dosya Seç")
        self.btn_pick.clicked.connect(self.pick_file)

        drop_row = QHBoxLayout()
        drop_row.addWidget(self.btn_pick)
        drop_row.addStretch()

        self.preview_box = QGroupBox("Önizleme (Sadece 5. Frame)")
        self.preview_layout = QVBoxLayout()
        self.preview_box.setLayout(self.preview_layout)

        self.preview_label = PreviewLabel()
        self.preview_label.setText("Önizleme yok")
        self.preview_layout.addWidget(self.preview_label)

        left = QVBoxLayout()
        left.addWidget(self.drop)
        left.addLayout(drop_row)
        left.addWidget(self.preview_box)
        left.addStretch()

        left_widget = QWidget()
        left_widget.setLayout(left)

        # --- Right: metadata
        self.meta_box = QGroupBox("Video Bilgileri")
        self.meta_form = QFormLayout()
        self.lbl_title = QLabel("-")
        self.lbl_fps = QLabel("-")
        self.lbl_total = QLabel("-")
        self.lbl_dur = QLabel("-")
        self.lbl_res = QLabel("-")
        self.meta_form.addRow("Başlık:", self.lbl_title)
        self.meta_form.addRow("FPS:", self.lbl_fps)
        self.meta_form.addRow("Toplam Frame:", self.lbl_total)
        self.meta_form.addRow("Süre (sn):", self.lbl_dur)
        self.meta_form.addRow("Çözünürlük:", self.lbl_res)
        self.meta_box.setLayout(self.meta_form)

        # --- Range
        self.range_box = QGroupBox("Kesme Aralığı")
        self.rb_frame = QRadioButton("Frame aralığı")
        self.rb_sec = QRadioButton("Zaman aralığı (sn)")
        self.rb_frame.setChecked(True)

        self.bg = QButtonGroup(self)
        self.bg.addButton(self.rb_frame)
        self.bg.addButton(self.rb_sec)
        self.rb_frame.toggled.connect(self.update_range_mode)

        self.sp_start_frame = QSpinBox()
        self.sp_end_frame = QSpinBox()
        self.sp_start_frame.setRange(0, 0)
        self.sp_end_frame.setRange(0, 0)

        self.ds_start = QDoubleSpinBox()
        self.ds_end = QDoubleSpinBox()
        self.ds_start.setDecimals(3)
        self.ds_end.setDecimals(3)
        self.ds_start.setRange(0.0, 0.0)
        self.ds_end.setRange(0.0, 0.0)

        self.btn_to_first = QPushButton("İlk’e Getir")
        self.btn_to_last = QPushButton("Son’a Getir")
        self.btn_to_first.clicked.connect(self.set_to_first)
        self.btn_to_last.clicked.connect(self.set_to_last)
        self.btn_to_first.setEnabled(False)
        self.btn_to_last.setEnabled(False)

        btn_row = QHBoxLayout()
        btn_row.addWidget(self.btn_to_first)
        btn_row.addWidget(self.btn_to_last)

        range_form = QFormLayout()
        range_form.addRow(self.rb_frame, QLabel(""))
        range_form.addRow("Start frame:", self.sp_start_frame)
        range_form.addRow("End frame:", self.sp_end_frame)
        range_form.addRow(self.rb_sec, QLabel(""))
        range_form.addRow("Start sn:", self.ds_start)
        range_form.addRow("End sn:", self.ds_end)

        range_wrap = QVBoxLayout()
        range_wrap.addLayout(range_form)
        range_wrap.addLayout(btn_row)
        self.range_box.setLayout(range_wrap)

        # --- Output
        self.sample_box = QGroupBox("Örnekleme / Çıkış")
        self.in_target_fps = QDoubleSpinBox()
        self.in_target_fps.setDecimals(3)
        self.in_target_fps.setRange(0.0, 10000.0)
        self.in_target_fps.setValue(0.0)
        self.in_target_fps.setToolTip("0: her kare")

        self.out_dir = QLineEdit()
        self.out_dir.setPlaceholderText(
            "Çıkış klasörü (boşsa video yanına 'frames_out')"
        )
        self.btn_out = QPushButton("Klasör Seç")
        self.btn_out.clicked.connect(self.pick_out_dir)

        out_row = QHBoxLayout()
        out_row.addWidget(self.out_dir)
        out_row.addWidget(self.btn_out)

        sample_form = QFormLayout()
        sample_form.addRow("Hedef FPS:", self.in_target_fps)

        sample_wrap = QVBoxLayout()
        sample_wrap.addLayout(sample_form)
        sample_wrap.addLayout(out_row)
        self.sample_box.setLayout(sample_wrap)

        # --- Extract
        self.btn_extract = QPushButton("Çıkart (Frames Kaydet)")
        self.btn_extract.clicked.connect(self.on_extract)
        self.btn_extract.setEnabled(False)

        self.progress = QProgressBar()
        self.progress.setValue(0)
        self.progress.setVisible(False)

        right = QVBoxLayout()
        right.addWidget(self.meta_box)
        right.addWidget(self.range_box)
        right.addWidget(self.sample_box)
        right.addWidget(self.btn_extract)
        right.addWidget(self.progress)
        right.addStretch()

        right_widget = QWidget()
        right_widget.setLayout(right)

        # --- Main layout (left bigger)
        main = QHBoxLayout()
        main.addWidget(left_widget, 3)
        main.addWidget(right_widget, 1)

        root = QWidget()
        root.setLayout(main)
        self.setCentralWidget(root)

        self.update_range_mode()

    # ---------- UI actions ----------

    def pick_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Video seç",
            "",
            "Video Files (*.mp4 *.avi *.mkv *.mov *.m4v);;All Files (*)",
        )
        if path:
            self.load_video(path)

    def pick_out_dir(self):
        d = QFileDialog.getExistingDirectory(self, "Çıkış klasörü seç")
        if d:
            self.out_dir.setText(d)

    def set_to_first(self):
        if not self.meta:
            return
        if self.rb_frame.isChecked():
            self.sp_start_frame.setValue(0)
        else:
            self.ds_start.setValue(0.0)

    def set_to_last(self):
        if not self.meta:
            return
        total = int(self.meta["total_frames"])
        dur = float(self.meta["duration_sec"])
        if self.rb_frame.isChecked():
            self.sp_end_frame.setValue(max(0, total - 1))
        else:
            self.ds_end.setValue(max(0.0, dur))

    def set_full_range_defaults(self):
        self.set_to_first()
        self.set_to_last()

    def load_video(self, path: str):
        try:
            if not os.path.isfile(path):
                raise ValueError("Dosya bulunamadı.")

            self.video_path = path
            self.meta = get_video_metadata(path)

            title = os.path.basename(path)
            fps = self.meta["fps"]
            total = self.meta["total_frames"]
            dur = self.meta["duration_sec"]
            res = self.meta["resolution"]

            self.lbl_title.setText(title)
            self.lbl_fps.setText(f"{fps:.3f}")
            self.lbl_total.setText(str(total))
            self.lbl_dur.setText(f"{dur:.3f}")
            self.lbl_res.setText(f"{res[0]} x {res[1]}")

            # Ranges
            self.sp_start_frame.setRange(0, max(0, total - 1))
            self.sp_end_frame.setRange(0, max(0, total - 1))
            self.ds_start.setRange(0.0, max(0.0, dur))
            self.ds_end.setRange(0.0, max(0.0, dur))

            self.set_full_range_defaults()

            # Preview: only 5th frame (index=4) or last if shorter
            frame, actual_idx = read_specific_frame(path, 4)
            pix = cv_frame_to_qpixmap(frame)
            self.preview_label.setToolTip(f"Önizleme Frame: {actual_idx}")
            self.preview_label.set_preview_pixmap(pix)

            self.btn_extract.setEnabled(True)
            self.btn_to_first.setEnabled(True)
            self.btn_to_last.setEnabled(True)
            self.drop.label.setText(
                f"✅ Yüklendi: {title}\nYeni video için sürükleyip bırakabilirsin."
            )

            # Default output dir suggestion
            if not self.out_dir.text().strip():
                base_dir = os.path.dirname(path)
                self.out_dir.setText(os.path.join(base_dir, "frames_out"))

        except Exception as e:
            self.video_path = None
            self.meta = None
            self.btn_extract.setEnabled(False)
            self.btn_to_first.setEnabled(False)
            self.btn_to_last.setEnabled(False)
            self.preview_label.setText("Önizleme yok")
            self.preview_label.set_preview_pixmap(QPixmap())
            QMessageBox.critical(self, "Hata", str(e))

    def update_range_mode(self):
        is_frame = self.rb_frame.isChecked()
        self.sp_start_frame.setEnabled(is_frame)
        self.sp_end_frame.setEnabled(is_frame)
        self.ds_start.setEnabled(not is_frame)
        self.ds_end.setEnabled(not is_frame)

    def on_extract(self):
        if not self.video_path or not self.meta:
            QMessageBox.warning(self, "Uyarı", "Önce bir video yükle.")
            return

        try:
            if self.rb_frame.isChecked():
                start_frame = int(self.sp_start_frame.value())
                end_frame = int(self.sp_end_frame.value())
                s, e = normalize_range(
                    self.meta,
                    mode="frames",
                    start_frame=start_frame,
                    end_frame=end_frame,
                )
            else:
                start_sec = float(self.ds_start.value())
                end_sec = float(self.ds_end.value())
                s, e = normalize_range(
                    self.meta, mode="seconds", start_sec=start_sec, end_sec=end_sec
                )

            out_dir = self.out_dir.text().strip()
            if not out_dir:
                base_dir = os.path.dirname(self.video_path)
                out_dir = os.path.join(base_dir, "frames_out")

            tfps = float(self.in_target_fps.value())
            if tfps <= 0:
                tfps = None

            params = ExtractParams(
                video_path=self.video_path,
                output_dir=out_dir,
                start_frame=s,
                end_frame=e,
                target_fps=tfps,
            )

            self.btn_extract.setEnabled(False)
            self.progress.setVisible(True)
            self.progress.setValue(0)

            self.worker = ExtractWorker(params)
            self.worker.progress.connect(self.progress.setValue)
            self.worker.finished_ok.connect(self.on_extract_done)
            self.worker.failed.connect(self.on_extract_failed)
            self.worker.start()

        except Exception as e:
            QMessageBox.critical(self, "Hata", str(e))

    def on_extract_done(self, saved_count: int, out_dir: str):
        self.btn_extract.setEnabled(True)
        self.progress.setVisible(False)
        QMessageBox.information(
            self, "Bitti", f"✅ Kaydedilen kare: {saved_count}\n📁 Klasör: {out_dir}"
        )

    def on_extract_failed(self, msg: str):
        self.btn_extract.setEnabled(True)
        self.progress.setVisible(False)
        QMessageBox.critical(self, "Hata", msg)


def main():
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
