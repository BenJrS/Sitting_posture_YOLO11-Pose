import os
import csv
import datetime
import threading
import time
import queue
import cv2
import customtkinter as ctk
from tkinter import filedialog, messagebox
from PIL import Image

from config import EXPORT_BASE_DIR, WEBCAM_WIDTH, WEBCAM_HEIGHT, GAZE_DEVICE
from video_sources import WebcamSource, IPCameraSource
from detectors import TiltDetector, PostureDetector
from gaze_wrapper import GazeEstimator

ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

class ProcessingThread(threading.Thread):
    def __init__(self, app_state, result_queue):
        super().__init__()
        self.app_state = app_state
        self.result_queue = result_queue
        self.stop_event = threading.Event()
        self.tilt_src = None; self.posture_src = None
        self.tilt_det = None; self.posture_det = None
        self.gaze_est = None
        self.csv_file = None; self.csv_writer = None
        self.frame_idx = 0

    def _create_src(self, type, val):
        try:
            if type == "Webcam": return WebcamSource(int(val), WEBCAM_WIDTH, WEBCAM_HEIGHT)
            else: return IPCameraSource(val)
        except Exception as e: raise RuntimeError(f"Lỗi Camera {val}: {e}")

    def run(self):
        try:
            if self.app_state['use_tilt']:
                self.tilt_det = TiltDetector(self.app_state['tilt_model'])
                self.tilt_src = self._create_src(self.app_state['tilt_type'], self.app_state['tilt_val'])
            if self.app_state['use_posture']:
                self.posture_det = PostureDetector(self.app_state['posture_model'])
                self.posture_src = self._create_src(self.app_state['posture_type'], self.app_state['posture_val'])
            if self.app_state['use_gaze']:
                self.gaze_est = GazeEstimator(device=GAZE_DEVICE)

            if self.app_state['logging']:
                now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                d = os.path.join(EXPORT_BASE_DIR, f"session_{now}")
                os.makedirs(d, exist_ok=True)
                path = os.path.join(d, "log_pro.csv")
                self.csv_file = open(path, "w", newline="", encoding="utf-8")
                self.csv_writer = csv.writer(self.csv_file)

                header = ["Timestamp", "Frame", "Tilt_Label", "Tilt_Conf"]
                kp_names = ["Nose", "L_Eye", "R_Eye", "L_Ear", "R_Ear", "L_Sho", "R_Sho"]
                for name in kp_names: header.extend([f"{name}_x", f"{name}_y"])
                header.extend(["Gaze_Label", "Pupil_L_x", "Pupil_L_y", "Pupil_R_x", "Pupil_R_y"])
                header.extend(["Posture_Label", "Box_x", "Box_y", "Box_w", "Box_h"])
                self.csv_writer.writerow(header)
                self.result_queue.put(("csv_path", path))
        except Exception as e:
            self.result_queue.put(("error", str(e))); return

        while not self.stop_event.is_set():
            start = time.time()
            f1_rgb, f2_rgb = None, None
            tilt_data = None; gaze_data = None; posture_data = None

            # CAM 1
            if self.tilt_src:
                ret, frame = self.tilt_src.read()
                if ret:
                    disp = frame.copy()
                    if self.frame_idx % 2 == 0:
                        res = self.tilt_det.infer(frame)
                        tilt_data = {"label": res.get('label'), "conf": res.get('confidence'), "kps": res.get('keypoints')}

                    if self.gaze_est and (self.frame_idx % 3 == 0):
                        res = self.gaze_est.infer(frame)
                        # Code gaze_wrapper moi se tra ve 'eyes_data'
                        gaze_data = {"label": res.get('label'), "eyes": res.get('eyes_data', [])}
                        if res.get('annotated') is not None: disp = res['annotated']

                    if tilt_data and tilt_data.get('label'):
                         cv2.putText(disp, f"T: {tilt_data['label']}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
                         if tilt_data['kps']:
                            for x,y in tilt_data['kps']: cv2.circle(disp, (int(x),int(y)), 4, (0,255,255), -1)

                    if gaze_data and gaze_data.get('label'):
                        col = (0, 255, 0) if "CENTER" in gaze_data['label'] else (0, 0, 255)
                        cv2.putText(disp, f"G: {gaze_data['label']}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, col, 2)

                    f1_rgb = cv2.cvtColor(disp, cv2.COLOR_BGR2RGB)

            # CAM 2
            if self.posture_src:
                ret, frame = self.posture_src.read()
                if ret:
                    disp = frame.copy()
                    if self.frame_idx % 3 == 0:
                        res = self.posture_det.infer(frame)
                        posture_data = {"label": res.get('label'), "bbox": res.get('bbox')}

                    if posture_data and posture_data.get('label'):
                        x, y, w, h = posture_data['bbox']
                        c = (255, 50, 50) if "bad" in posture_data['label'].lower() else (50, 255, 50)
                        cv2.rectangle(disp, (int(x-w/2), int(y-h/2)), (int(x+w/2), int(y+h/2)), c, 2)

                    f2_rgb = cv2.cvtColor(disp, cv2.COLOR_BGR2RGB)

            self.result_queue.put(("frames", (f1_rgb, f2_rgb)))
            self.result_queue.put(("data_update", {"tilt": tilt_data, "gaze": gaze_data, "posture": posture_data}))

            if self.csv_writer:
                ts = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]
                t_lbl = tilt_data['label'] if tilt_data else ""
                t_conf = tilt_data['conf'] if tilt_data else 0
                row = [ts, self.frame_idx, t_lbl, t_conf]
                t_kps = tilt_data['kps'] if (tilt_data and tilt_data.get('kps')) else []
                for i in range(7):
                    if i < len(t_kps): row.extend([t_kps[i][0], t_kps[i][1]])
                    else: row.extend([0, 0])

                g_lbl = gaze_data['label'] if gaze_data else ""
                row.append(g_lbl)
                g_eyes = gaze_data['eyes'] if (gaze_data and gaze_data.get('eyes')) else []
                if len(g_eyes) >= 1: row.extend([g_eyes[0]['rel'][0], g_eyes[0]['rel'][1]])
                else: row.extend([0, 0])
                if len(g_eyes) >= 2: row.extend([g_eyes[1]['rel'][0], g_eyes[1]['rel'][1]])
                else: row.extend([0, 0])

                p_lbl = posture_data['label'] if posture_data else ""
                row.append(p_lbl)
                if posture_data and posture_data.get('bbox'):
                    bx, by, bw, bh = posture_data['bbox']
                    row.extend([bx, by, bw, bh])
                else: row.extend([0, 0, 0, 0])

                self.csv_writer.writerow(row)

            self.frame_idx += 1
            if time.time() - start < 0.03: time.sleep(0.03 - (time.time() - start))

        if self.tilt_src: self.tilt_src.release()
        if self.posture_src: self.posture_src.release()
        if self.csv_file: self.csv_file.close()

    def stop(self):
        self.stop_event.set()
        self.join()

class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("GUI Sitting posture")
        self.geometry("1400x800")
        self.protocol("WM_DELETE_WINDOW", self.on_close)

        self.grid_columnconfigure(0, weight=0)
        self.grid_columnconfigure(1, weight=1)
        self.grid_columnconfigure(2, weight=0)
        self.grid_rowconfigure(0, weight=1)

        self.last_tilt = {"label": None, "time": 0}
        self.last_gaze = {"label": None, "time": 0}
        self.last_post = {"label": None, "time": 0}
        self.PERSIST_TIME = 1.0

        self._setup_ui()
        self.thread = None; self.queue = queue.Queue(); self.logging_active = False

    def _setup_ui(self):
        self.tilt_path = ""; self.post_path = ""
        self.tilt_type = ctk.StringVar(value="Webcam"); self.tilt_val = ctk.StringVar(value="0")
        self.post_type = ctk.StringVar(value="Webcam"); self.post_val = ctk.StringVar(value="1")

        # SIDEBAR
        self.sidebar = ctk.CTkFrame(self, width=250, corner_radius=0)
        self.sidebar.grid(row=0, column=0, sticky="nsew")
        self.sidebar.grid_propagate(False)
        ctk.CTkLabel(self.sidebar, text="CONTROL", font=("Arial", 16, "bold")).pack(pady=20)

        self._create_group(self.sidebar, "1. Nghiêng ngã & Gaze", self._sel_tilt, "tilt")
        self._create_group(self.sidebar, "2. Posture", self._sel_post, "post")

        sf = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        sf.pack(pady=10, fill="x", padx=10)
        self.sw_tilt = ctk.CTkSwitch(sf, text="Nghiêng ngã"); self.sw_tilt.select(); self.sw_tilt.pack(anchor="w")
        self.sw_gaze = ctk.CTkSwitch(sf, text="Gaze tracking"); self.sw_gaze.select(); self.sw_gaze.pack(anchor="w")
        self.sw_post = ctk.CTkSwitch(sf, text="Posture"); self.sw_post.select(); self.sw_post.pack(anchor="w")

        self.btn_run = ctk.CTkButton(self.sidebar, text="BẮT ĐẦU", fg_color="green", height=40, command=self.toggle_run)
        self.btn_run.pack(pady=20, padx=10, fill="x")

        self.btn_csv = ctk.CTkButton(self.sidebar, text="GHI LOG CSV (TẮT)", fg_color="#555", command=self.toggle_csv)
        self.btn_csv.pack(pady=5, padx=10, fill="x")

        # VIDEO
        # --- KHÔI PHỤC LẠI self.view ---
        self.view = ctk.CTkFrame(self, fg_color="#111")
        self.view.grid(row=0, column=1, sticky="nsew", padx=5, pady=5)
        self.view.rowconfigure(0, weight=1); self.view.rowconfigure(1, weight=1); self.view.columnconfigure(0, weight=1)
        # --------------------------------

        self.f1 = ctk.CTkLabel(self.view, text="CAM 1", fg_color="black")  # Có text ban đầu
        self.f1.grid(row=0, column=0, sticky="nsew", padx=2, pady=2)

        self.f2 = ctk.CTkLabel(self.view, text="CAM 2", fg_color="black")  # Có text ban đầu
        self.f2.grid(row=1, column=0, sticky="nsew", padx=2, pady=2)

        # RIGHT PANEL
        self.panel = ctk.CTkFrame(self, width=330, corner_radius=0)
        self.panel.grid(row=0, column=2, sticky="nsew")
        self.panel.grid_propagate(False)
        ctk.CTkLabel(self.panel, text="THÔNG SỐ", font=("Arial", 16, "bold")).pack(pady=20)

        FONT_DATA = ("Consolas", 13)
        FONT_TITLE = ("Arial", 12, "bold")

        # TILT
        self.box_tilt = ctk.CTkScrollableFrame(self.panel, fg_color="#222", height=200, label_text="NGHIÊNG NGÃ", label_fg_color="#4cc9f0", label_font=FONT_TITLE)
        self.box_tilt.pack(pady=5, padx=10, fill="x")
        self.lbl_tilt_st = ctk.CTkLabel(self.box_tilt, text="Label: --", font=FONT_DATA, anchor="w")
        self.lbl_tilt_st.pack(fill="x", padx=5)

        self.kps_labels = []
        names = ["Nose", "L-Eye", "R-Eye", "L-Ear", "R-Ear", "L-Sho", "R-Sho"]
        for name in names:
            l = ctk.CTkLabel(self.box_tilt, text=f"{name}: (----, ----)", font=FONT_DATA, anchor="w")
            l.pack(fill="x", padx=5)
            self.kps_labels.append(l)

        # GAZE
        self.box_gaze = ctk.CTkFrame(self.panel, fg_color="#222", height=150)
        self.box_gaze.pack(pady=5, padx=10, fill="x")
        self.box_gaze.pack_propagate(False)
        ctk.CTkLabel(self.box_gaze, text="GAZE TRACKING", text_color="#4cc9f0", font=FONT_TITLE).pack(anchor="w", padx=5)
        self.lbl_gaze_st = ctk.CTkLabel(self.box_gaze, text="Label: --", font=("Consolas", 16, "bold"), text_color="yellow", anchor="w")
        self.lbl_gaze_st.pack(fill="x", padx=5)
        self.lbl_eye_L = ctk.CTkLabel(self.box_gaze, text="L-Eye: (---, ---)", font=FONT_DATA, anchor="w")
        self.lbl_eye_L.pack(fill="x", padx=5)
        self.lbl_eye_R = ctk.CTkLabel(self.box_gaze, text="R-Eye: (---, ---)", font=FONT_DATA, anchor="w")
        self.lbl_eye_R.pack(fill="x", padx=5)

        # POSTURE
        self.box_post = ctk.CTkFrame(self.panel, fg_color="#222", height=100)
        self.box_post.pack(pady=5, padx=10, fill="x")
        self.box_post.pack_propagate(False)
        ctk.CTkLabel(self.box_post, text="POSTURE QUALITY", text_color="#4cc9f0", font=FONT_TITLE).pack(anchor="w", padx=5)
        self.lbl_post_st = ctk.CTkLabel(self.box_post, text="Status: --", font=FONT_DATA, anchor="w")
        self.lbl_post_st.pack(fill="x", padx=5)
        self.lbl_post_bb = ctk.CTkLabel(self.box_post, text="BBox: [---, ---, ---, ---]", font=FONT_DATA, anchor="w")
        self.lbl_post_bb.pack(fill="x", padx=5)

    def _create_group(self, parent, title, cmd, prefix):
        f = ctk.CTkFrame(parent, fg_color="#333")
        f.pack(pady=5, padx=10, fill="x")
        ctk.CTkLabel(f, text=title).pack()
        btn = ctk.CTkButton(f, text="Chọn Model", command=cmd, height=25)
        btn.pack(pady=5, padx=5, fill="x")
        setattr(self, f"btn_sel_{prefix}", btn)
        ctk.CTkOptionMenu(f, values=["Webcam", "IP"], variable=getattr(self, f"{prefix}_type")).pack(pady=2, padx=5, fill="x")
        ctk.CTkEntry(f, textvariable=getattr(self, f"{prefix}_val")).pack(pady=2, padx=5, fill="x")

    def _sel_tilt(self):
        p = filedialog.askopenfilename();
        if p: self.tilt_path = p; self.btn_sel_tilt.configure(fg_color="blue", text="Đã chọn")
    def _sel_post(self):
        p = filedialog.askopenfilename();
        if p: self.post_path = p; self.btn_sel_post.configure(fg_color="blue", text="Đã chọn")

    def toggle_csv(self):
        self.logging_active = not self.logging_active
        if self.logging_active: self.btn_csv.configure(text="ĐANG GHI LOG...", fg_color="red")
        else: self.btn_csv.configure(text="GHI LOG CSV (TẮT)", fg_color="#555")

    def toggle_run(self):
        if self.thread and self.thread.is_alive():
            self.thread.stop(); self.thread = None; self.btn_run.configure(text="BẮT ĐẦU", fg_color="green")

            self.f1.configure(image=None, text="CAM 1")
            self.f2.configure(image=None, text="CAM 2")

        else:
            cfg = {
                'use_tilt': self.sw_tilt.get(), 'tilt_model': self.tilt_path, 'tilt_type': self.tilt_type.get(), 'tilt_val': self.tilt_val.get(),
                'use_posture': self.sw_post.get(), 'posture_model': self.post_path, 'posture_type': self.post_type.get(), 'posture_val': self.post_val.get(),
                'use_gaze': self.sw_gaze.get(), 'logging': self.logging_active
            }
            if cfg['use_tilt'] and not cfg['tilt_model']: return messagebox.showerror("Lỗi", "Thiếu model Tilt")
            if cfg['use_posture'] and not cfg['posture_model']: return messagebox.showerror("Lỗi", "Thiếu model Posture")

            self.thread = ProcessingThread(cfg, self.queue)
            self.thread.start()
            self.btn_run.configure(text="DỪNG", fg_color="red")
            self.check()

    def check(self):
        if not self.thread: return
        try:
            while True:
                t, c = self.queue.get_nowait()
                if t == "frames":
                    i1, i2 = c
                    if i1 is not None:
                        self.f1.configure(image=ctk.CTkImage(Image.fromarray(i1), size=(640, 360)), text="")

                    if i2 is not None:
                        self.f2.configure(image=ctk.CTkImage(Image.fromarray(i2), size=(640, 360)), text="")
                elif t == "data_update":
                    self._update_panel_safe(c)
                elif t == "error":
                    messagebox.showerror("Error", c); self.toggle_run()
                elif t == "csv_path":
                    messagebox.showinfo("CSV", f"Lưu log tại:\n{c}")
        except queue.Empty:
            pass
        self.after(20, self.check)

    def _update_panel_safe(self, data):
        curr_time = time.time()

        # --- TILT ---
        t_raw = data.get('tilt')
        if t_raw and t_raw.get('label'): self.last_tilt = {"data": t_raw, "time": curr_time}

        if (curr_time - self.last_tilt["time"]) < self.PERSIST_TIME and self.last_tilt.get("data"):
            d = self.last_tilt["data"]
            self.lbl_tilt_st.configure(text=f"Label: {d['label']} ({d['conf']:.2f})")
            kps = d.get('kps', [])
            names = ["Nose", "L-Eye", "R-Eye", "L-Ear", "R-Ear", "L-Sho", "R-Sho"]
            for i, label_widget in enumerate(self.kps_labels):
                name_str = names[i] if i < len(names) else f"KP-{i}"
                if i < len(kps):
                    x, y = kps[i]
                    label_widget.configure(text=f"{name_str}: ({int(x):04d}, {int(y):04d})")
                else:
                    label_widget.configure(text=f"{name_str}: (----, ----)")
        else:
            self.lbl_tilt_st.configure(text="Label: Searching...")

        # --- GAZE ---
        g_raw = data.get('gaze')
        if g_raw and g_raw.get('label'): self.last_gaze = {"data": g_raw, "time": curr_time}

        if (curr_time - self.last_gaze["time"]) < self.PERSIST_TIME and self.last_gaze.get("data"):
            d = self.last_gaze["data"]
            self.lbl_gaze_st.configure(text=f"Dir: {d['label'].upper()}")
            eyes = d.get('eyes', [])
            if len(eyes) >= 1:
                lx, ly = eyes[0]['rel']
                self.lbl_eye_L.configure(text=f"L-Eye: ({lx:04d}, {ly:04d})")
            if len(eyes) >= 2:
                rx, ry = eyes[1]['rel']
                self.lbl_eye_R.configure(text=f"R-Eye: ({rx:04d}, {ry:04d})")
        else:
            self.lbl_gaze_st.configure(text="Dir: --")
            self.lbl_eye_L.configure(text="L-Eye: (--, --)")
            self.lbl_eye_R.configure(text="R-Eye: (--, --)")

        # --- POSTURE ---
        p_raw = data.get('posture')
        if p_raw and p_raw.get('label'): self.last_post = {"data": p_raw, "time": curr_time}

        if (curr_time - self.last_post["time"]) < self.PERSIST_TIME and self.last_post.get("data"):
            d = self.last_post["data"]
            self.lbl_post_st.configure(text=f"Status: {d['label']}")
            if d.get('bbox'):
                x, y, w, h = d['bbox']
                self.lbl_post_bb.configure(text=f"BBox: [{int(x):03d}, {int(y):03d}, {int(w):03d}, {int(h):03d}]")
        else:
            self.lbl_post_st.configure(text="Status: Searching...")

    def on_close(self):
        if self.thread: self.thread.stop()
        self.destroy()