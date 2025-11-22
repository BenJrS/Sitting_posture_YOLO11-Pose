from typing import Any, Dict
import cv2
import numpy as np
import mediapipe as mp

# MediaPipe Face Mesh Constants
mp_face_mesh = mp.solutions.face_mesh
LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]
LEFT_IRIS = [474, 475, 476, 477]
RIGHT_IRIS = [469, 470, 471, 472]


class GazeEstimator:
    def __init__(self, device: str = "cpu"):
        self.face_mesh = mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self._available = True

    def _euclidean_distance(self, point1, point2):
        return np.linalg.norm(point1 - point2)

    def _get_blink_ratio(self, eye_points, landmarks):
        p1 = landmarks[eye_points[0]]
        p2 = landmarks[eye_points[1]]
        p3 = landmarks[eye_points[2]]
        p4 = landmarks[eye_points[3]]
        p5 = landmarks[eye_points[4]]
        p6 = landmarks[eye_points[5]]

        ear_v = (self._euclidean_distance(p2, p6) + self._euclidean_distance(p3, p5)) / 2.0
        ear_h = self._euclidean_distance(p1, p4)
        return ear_v / (ear_h + 1e-6)

    def _get_gaze_ratio(self, eye_points, iris_center, landmarks):
        inner = landmarks[eye_points[0]]
        outer = landmarks[eye_points[3]]
        eye_width = self._euclidean_distance(inner, outer)
        dist_to_inner = self._euclidean_distance(iris_center, inner)
        if eye_width == 0: return 0.5
        return dist_to_inner / eye_width

    def infer(self, frame_bgr) -> Dict[str, Any]:
        if frame_bgr is None: return {"status": "no_frame"}

        h, w = frame_bgr.shape[:2]
        rgb_frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)

        label = "unknown"
        blink = False
        annotated = frame_bgr.copy()
        eyes_data = []  # <--- KHỞI TẠO LIST CHỨA TOẠ ĐỘ

        if results.multi_face_landmarks:
            mesh_points = np.array(
                [np.multiply([p.x, p.y], [w, h]).astype(int) for p in results.multi_face_landmarks[0].landmark])

            # 1. Tính Blink Ratio
            left_ear = self._get_blink_ratio(LEFT_EYE, mesh_points)
            right_ear = self._get_blink_ratio(RIGHT_EYE, mesh_points)
            avg_ear = (left_ear + right_ear) / 2.0

            if avg_ear < 0.15:
                blink = True
                label = "blinking"
            else:
                # 2. Tính Gaze Direction & Toạ độ
                (l_cx, l_cy), l_radius = cv2.minEnclosingCircle(mesh_points[LEFT_IRIS])
                (r_cx, r_cy), r_radius = cv2.minEnclosingCircle(mesh_points[RIGHT_IRIS])

                center_left = np.array([l_cx, l_cy], dtype=np.int32)
                center_right = np.array([r_cx, r_cy], dtype=np.int32)

                # --- QUAN TRỌNG: Lưu toạ độ vào eyes_data để UI lấy ---
                # Cấu trúc dictionary này khớp với cách ui_app.py đang gọi: eyes[0]['rel']
                eyes_data = [
                    {"rel": (int(l_cx), int(l_cy))},  # Mắt trái (Left)
                    {"rel": (int(r_cx), int(r_cy))}  # Mắt phải (Right)
                ]
                # -------------------------------------------------------

                # Vẽ visual
                cv2.circle(annotated, center_left, int(l_radius), (0, 255, 0), 1, cv2.LINE_AA)
                cv2.circle(annotated, center_right, int(r_radius), (0, 255, 0), 1, cv2.LINE_AA)

                gaze_l = self._get_gaze_ratio(LEFT_EYE, center_left, mesh_points)
                gaze_r = self._get_gaze_ratio(RIGHT_EYE, center_right, mesh_points)
                avg_gaze = (gaze_l + gaze_r) / 2.0

                if avg_gaze < 0.45:
                    label = "right"
                elif avg_gaze > 0.55:
                    label = "left"
                else:
                    label = "center"
        else:
            return {"status": "no_face", "label": "no_face", "annotated": annotated, "eyes_data": []}

        return {
            "status": "ok",
            "label": label,
            "blink": blink,
            "annotated": annotated,
            "eyes_data": eyes_data  # <--- TRẢ VỀ DỮ LIỆU MỚI
        }