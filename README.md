# GUI Sitting Posture & Gaze Monitor

Ứng dụng GUI theo dõi tư thế ngồi, nghiêng/ngã và hướng nhìn (gaze) theo thời gian thực từ 2 camera (hỗ trợ cả webcam và IP camera từ điện thoại).

---

## Tính năng chính
- 🎥 Hai luồng video song song
  - CAM 1: Nghiêng/ngã + Gaze.
  - CAM 2: Tư thế ngồi tổng thể (good/bad.
- 🧠 TiltDetector (nghiêng/ngã)
  - Dựa trên Ultralytics YOLO.
  - Trích xuất 7 keypoint: Nose, L/R Eye, L/R Ear, L/R Shoulder.
  - Hiển thị label + confidence + vẽ keypoint trên CAM 1.
- 👀 GazeEstimator (gaze tracking)
  - Dùng MediaPipe Face Mesh.
  - Nhận diện hướng nhìn: `left` / `right` / `center` / `blinking` / `no_face`.
  - Phát hiện nháy mắt (blink) dựa trên EAR (Eye Aspect Ratio).
  - Trả về tọa độ tương đối đồng tử 2 mắt.
- 🪑 PostureDetector (tư thế ngồi)
  - Dựa trên YOLOv5 (thư viện `yolov5`).
  - Phân loại posture (label theo dataset của bạn).
  - Vẽ bounding box: thường là box xanh cho tư thế tốt, đỏ cho tư thế xấu (tùy cách train/label).

---

## Yêu cầu hệ thống

### 1) Phần cứng
- CPU: PC / laptop phổ biến.
- GPU (tùy chọn):
  - Không bắt buộc — app vẫn chạy ổn trên CPU (đã test với MacBook Air M1).
  - Nếu dùng GPU: cài bản PyTorch tương thích với CUDA.
- Camera:
  - 1–2 webcam (rời hoặc built-in) hoặc 1–2 IP camera từ điện thoại (Android/iOS) qua cùng mạng Wi‑Fi.

### 2) Phần mềm
- Python 3.10+ (khuyên dùng 3.10 / 3.11).
- Hệ điều hành:
  - Windows 10/11, macOS (Apple Silicon / Intel), Linux (Ubuntu, …).

### 3) Thư viện Python (requirements.txt)
```text
ultralytics>=8.0.0
yolov5>=7.0.0
opencv-python
numpy
Pillow
torch
torchvision
customtkinter
packaging
mediapipe>=0.10.0
```
⚠️ Lưu ý: `yolov5` ở đây là thư viện Python của repo YOLOv5 (pip cài trực tiếp từ PyPI hoặc clone repo). Cần mạng để pip cài về.

---

## Cài đặt

1. Tạo môi trường ảo (khuyên dùng)
```bash
python -m venv .venv

# macOS / Linux
source .venv/bin/activate

# Windows
.venv\Scripts\activate
```

2. Cập nhật pip và cài dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

- Nếu muốn dùng GPU: hãy cài `torch` / `torchvision` theo hướng dẫn chính thức của PyTorch, tương thích với phiên bản CUDA trên máy.
- Nếu gặp lỗi khi cài `yolov5`, có thể clone repo YOLOv5 và cài thủ công:
```bash
git clone https://github.com/ultralytics/yolov5.git
cd yolov5
pip install -r requirements.txt
```

---

## Cách chạy

Từ thư mục chứa project:
```bash
python main.py
```

Nếu cài đặt thành công, cửa sổ GUI sẽ hiện ra với:
- Bên trái: Sidebar CONTROL.
- Ở giữa: 2 khung video CAM 1 và CAM 2.
- Bên phải: Panel thông số của Tilt / Gaze / Posture.

---

## Giao diện & Điều khiển

- Sidebar:
  - Chọn model `.pt` cho TiltDetector và PostureDetector.
  - Chọn nguồn Video (Webcam index hoặc IP camera URL)
      Với Webcam nhập index 0,1
      Với IP camera URL nhập http://192.168.x.x:port/video
  - Toggle cho từng nhánh: Tilt, Gaze, Posture.
  - Nút BẮT ĐẦU / Dừng luồng.
  - Nút GHI LOG CSV (bật/tắt ghi tay).
- CAM 1:
  - Hiển thị keypoint (7 điểm), label tilt + confidence.
  - Hiển thị overlay gaze (hướng nhìn) và trạng thái nháy mắt.
- CAM 2:
  - Hiển thị bounding box posture kèm label và confidence.
  - Màu box tuỳ theo label (ví dụ: xanh = good, đỏ = bad).
- Panel phải:
  - Hiển thị chi tiết numeric: tọa độ keypoint, EAR, tọa độ đồng tử, label posture + confidence, thời gian frame.


