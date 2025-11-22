GUI Sitting Posture & Gaze Monitor

Ứng dụng GUI theo dõi tư thế ngồi, nghiêng/ngã và hướng nhìn (gaze) theo thời gian thực từ 2 camera
(hỗ trợ cả webcam và IP camera từ điện thoại).

Phù hợp cho:
	•	Demo / báo cáo đề tài HAR – posture/gaze.
	•	Thu thập log dữ liệu (CSV) để phân tích / train thêm mô hình.
	•	Giám sát tư thế học tập/làm việc.

⸻

1. Tính năng chính
	•	🎥 Hai luồng video song song
	•	CAM 1: Nghiêng/ngã đầu + Gaze.
	•	CAM 2: Tư thế ngồi tổng thể (good/bad/…).
	•	🧠 TiltDetector (nghiêng/ngã)
	•	Dựa trên Ultralytics YOLO.
	•	Trích xuất 7 keypoint: Nose, L/R Eye, L/R Ear, L/R Shoulder.
	•	Hiển thị label + confidence + vẽ keypoint trên CAM 1.
	•	👀 GazeEstimator (gaze tracking)
	•	Dùng MediaPipe Face Mesh.
	•	Nhận diện:
	•	Hướng nhìn: left / right / center / blinking / no_face.
	•	Nháy mắt (blink) dựa trên EAR.
	•	Trả về tọa độ tương đối đồng tử 2 mắt.
	•	🪑 PostureDetector (tư thế ngồi)
	•	Dựa trên YOLOv5 (thư viện yolov5).
	•	Phân loại posture (label theo dataset của bạn).
	•	Vẽ bounding box:
	•	Thường: box xanh với tư thế tốt, đỏ với tư thế xấu (tùy cách bạn train/label).
	•	📊 Giao diện CustomTkinter
	•	Sidebar điều khiển:
	•	Chọn model .pt cho nghiêng/ngã và posture.
	•	Chọn nguồn video: Webcam hoặc IP.
	•	Bật/tắt từng nhánh: Nghiêng ngã, Gaze tracking, Posture.
	•	Nút BẮT ĐẦU / dừng.
	•	Nút GHI LOG CSV.
	•	Panel phải hiển thị:
	•	Label tilt + 7 keypoint.
	•	Label gaze + thông tin mắt.
	•	Label posture + bounding box.
	•	📁 Ghi log CSV tự động
	•	Lưu vào: exports/session_YYYYMMDD_HHMMSS/log_pro.csv.
	•	Gồm đầy đủ thời gian, tilt/gaze/posture, keypoint & bounding box.

⸻

2. Yêu cầu hệ thống

2.1. Phần cứng
	•	CPU: PC/laptop phổ biến.
	•	GPU (tùy chọn):
	•	Không bắt buộc – app vẫn chạy ổn trên CPU (đã test với MacBook Air M1).
	•	Nếu dùng GPU:
	•	Cài bản PyTorch tương thích CUDA.
	•	Camera:
	•	1–2 webcam (rời hoặc built-in).
	•	Hoặc 1–2 IP camera từ điện thoại (Android/iOS) qua cùng mạng Wi-Fi.

2.2. Phần mềm
	•	Python 3.10+ (khuyên dùng 3.10/3.11).
	•	Hệ điều hành:
	•	✅ Windows 10/11
	•	✅ macOS (Apple Silicon / Intel)
	•	✅ Linux (Ubuntu, …)

2.3. Thư viện Python (trong requirements.txt)
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
⚠️ Lưu ý: yolov5 ở đây là thư viện Python của repo YOLOv5, cần mạng để pip cài về.

3. Cài đặt

3.1. Tạo môi trường ảo (khuyên dùng)
python -m venv .venv

# macOS / Linux
source .venv/bin/activate

# Windows
.venv\Scripts\activate

3.2. Cài dependencies
pip install --upgrade pip
pip install -r requirements.txt
Nếu cài torch/torchvision cho GPU, hãy làm theo hướng dẫn chính thức của PyTorch và đảm bảo version tương thích với CUDA.

4. Cách chạy

Từ thư mục chứa project:
python main.py
Nếu cài đặt thành công, cửa sổ GUI sẽ hiện ra với:
	•	Bên trái: Sidebar CONTROL.
	•	Ở giữa: 2 khung video CAM 1 và CAM 2.
	•	Bên phải: panel thông số của Tilt/Gaze/Posture.
