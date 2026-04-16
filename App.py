"""
AI Shape Detector - Stylish GUI
Requires: opencv-python, mediapipe, numpy, Pillow, tensorflow
Run: python gui_app.py
"""

import tkinter as tk
from tkinter import ttk, font
import cv2
import numpy as np
import mediapipe as mp
from PIL import Image, ImageTk
import threading
import time
import math


# ─────────────────────────────────────────────────────────────
#  PALETTE  (dark sci-fi / holographic)
# ─────────────────────────────────────────────────────────────
BG_DARK       = "#0A0C12"
BG_PANEL      = "#10141E"
BG_CARD       = "#151A27"
ACCENT_CYAN   = "#00E5FF"
ACCENT_PURPLE = "#B388FF"
ACCENT_GREEN  = "#00FF9D"
ACCENT_PINK   = "#FF4081"
TEXT_PRIMARY  = "#E8EAED"
TEXT_MUTED    = "#607B96"
BORDER_COLOR  = "#1E2D45"
GLOW_CYAN     = "#00B8D9"


# ─────────────────────────────────────────────────────────────
#  HAND TRACKER  (MediaPipe)
# ─────────────────────────────────────────────────────────────
class HandTracker:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.6
        )
        self.mp_draw = mp.solutions.drawing_utils
        self.draw_spec_dot  = self.mp_draw.DrawingSpec(color=(0,229,255), thickness=-1, circle_radius=4)
        self.draw_spec_line = self.mp_draw.DrawingSpec(color=(179,136,255), thickness=2)

    def find_hand(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.hands.process(rgb)
        landmarks = []
        if result.multi_hand_landmarks:
            for hl in result.multi_hand_landmarks:
                self.mp_draw.draw_landmarks(
                    frame, hl,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.draw_spec_dot,
                    self.draw_spec_line
                )
                for lm in hl.landmark:
                    h, w = frame.shape[:2]
                    landmarks.append((int(lm.x * w), int(lm.y * h)))
        return frame, landmarks


# ─────────────────────────────────────────────────────────────
#  AIR DRAWER
# ─────────────────────────────────────────────────────────────
class AirDrawer:
    def __init__(self, w=640, h=480):
        self.w, self.h = w, h
        self.canvas = np.zeros((h, w, 3), dtype=np.uint8)
        self.points = []
        self.color  = (0, 229, 255)   # cyan default
        self.thickness = 5

    def update(self, landmarks, drawing=True):
        if landmarks and drawing:
            x, y = landmarks[8]
            self.points.append((x, y))
            for i in range(1, len(self.points)):
                t = i / max(len(self.points), 1)
                # gradient: cyan → purple
                r = int(0   + t * 179)
                g = int(229 - t * 93)
                b = int(255)
                cv2.line(self.canvas,
                         self.points[i-1], self.points[i],
                         (r, g, b), self.thickness)
        return self.canvas

    def clear(self):
        self.canvas = np.zeros((self.h, self.w, 3), dtype=np.uint8)
        self.points = []

    def get_point_count(self):
        return len(self.points)


# ─────────────────────────────────────────────────────────────
#  SHAPE CLASSIFIER  (geometry heuristics — no model needed)
# ─────────────────────────────────────────────────────────────
SHAPES = ["circle", "triangle", "square", "rectangle",
          "pentagon", "hexagon", "unknown"]

SHAPE_COLORS = {
    "circle"    : ACCENT_CYAN,
    "triangle"  : ACCENT_PINK,
    "square"    : ACCENT_GREEN,
    "rectangle" : ACCENT_GREEN,
    "pentagon"  : ACCENT_PURPLE,
    "hexagon"   : "#FFD54F",
    "unknown"   : TEXT_MUTED,
}

def classify_shape(canvas_bgr):
    """Classify drawn shape using contour approximation."""
    gray = cv2.cvtColor(canvas_bgr, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)
    kernel = np.ones((5, 5), np.uint8)
    thresh = cv2.dilate(thresh, kernel, iterations=2)
    thresh = cv2.erode(thresh, kernel, iterations=1)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return "unknown", 0.0

    c = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(c)
    if area < 400:
        return "unknown", 0.0

    peri   = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.035 * peri, True)
    verts  = len(approx)

    # circularity
    circularity = 4 * math.pi * area / (peri * peri) if peri > 0 else 0

    if circularity > 0.75:
        return "circle", round(min(circularity, 1.0), 2)

    # bounding box aspect ratio
    x, y, w, h = cv2.boundingRect(c)
    aspect = w / h if h > 0 else 1

    if verts == 3:
        return "triangle", 0.92
    elif verts == 4:
        if 0.85 <= aspect <= 1.15:
            return "square", 0.94
        else:
            return "rectangle", 0.90
    elif verts == 5:
        return "pentagon", 0.88
    elif verts >= 6:
        if circularity > 0.65:
            return "circle", 0.85
        return "hexagon", 0.86
    else:
        return "unknown", 0.50


# ─────────────────────────────────────────────────────────────
#  ANIMATED CANVAS WIDGET  (decorative grid / pulse)
# ─────────────────────────────────────────────────────────────
class AnimatedBG(tk.Canvas):
    def __init__(self, master, **kw):
        super().__init__(master, bg=BG_DARK, highlightthickness=0, **kw)
        self._tick = 0
        self._lines = []
        self.bind("<Configure>", self._on_resize)
        self._animate()

    def _on_resize(self, e):
        self.delete("grid")
        self._lines = []
        step = 40
        for x in range(0, e.width, step):
            lid = self.create_line(x, 0, x, e.height,
                                   fill=BORDER_COLOR, tags="grid", width=1)
            self._lines.append(lid)
        for y in range(0, e.height, step):
            lid = self.create_line(0, y, e.width, y,
                                   fill=BORDER_COLOR, tags="grid", width=1)
            self._lines.append(lid)

    def _animate(self):
        self._tick += 1
        alpha = int(20 + 8 * math.sin(self._tick * 0.05))
        col = f"#{alpha:02x}{alpha+5:02x}{alpha+10:02x}"
        for lid in self._lines:
            self.itemconfig(lid, fill=col)
        self.after(50, self._animate)


# ─────────────────────────────────────────────────────────────
#  CONFIDENCE RING WIDGET
# ─────────────────────────────────────────────────────────────
class ConfidenceRing(tk.Canvas):
    def __init__(self, master, size=120, **kw):
        super().__init__(master, width=size, height=size,
                         bg=BG_PANEL, highlightthickness=0, **kw)
        self.size = size
        self.value = 0.0
        self.color = ACCENT_CYAN
        self._draw(0, ACCENT_CYAN)

    def set(self, value, color):
        self.value = value
        self.color = color
        self._draw(value, color)

    def _draw(self, v, col):
        self.delete("all")
        s = self.size
        pad = 12
        # track
        self.create_arc(pad, pad, s-pad, s-pad,
                        start=90, extent=-360,
                        outline=BORDER_COLOR, width=8, style=tk.ARC)
        # fill
        extent = -int(v * 360)
        if extent != 0:
            self.create_arc(pad, pad, s-pad, s-pad,
                            start=90, extent=extent,
                            outline=col, width=8, style=tk.ARC)
        # percentage text
        pct = f"{int(v*100)}%"
        self.create_text(s//2, s//2,
                         text=pct,
                         fill=col,
                         font=("Courier New", 16, "bold"))


# ─────────────────────────────────────────────────────────────
#  HISTORY LOG ENTRY
# ─────────────────────────────────────────────────────────────
class HistoryEntry(tk.Frame):
    def __init__(self, master, shape, conf, idx, **kw):
        super().__init__(master, bg=BG_CARD,
                         highlightbackground=BORDER_COLOR,
                         highlightthickness=1, **kw)
        col = SHAPE_COLORS.get(shape, TEXT_MUTED)

        # number badge
        badge = tk.Label(self, text=f"#{idx:02d}",
                         bg=BG_DARK, fg=TEXT_MUTED,
                         font=("Courier New", 9), width=4,
                         padx=4, pady=2)
        badge.pack(side=tk.LEFT)

        # dot
        dot = tk.Canvas(self, width=10, height=10,
                        bg=BG_CARD, highlightthickness=0)
        dot.create_oval(2, 2, 8, 8, fill=col, outline="")
        dot.pack(side=tk.LEFT, padx=(4, 0))

        # shape name
        tk.Label(self, text=shape.upper(),
                 bg=BG_CARD, fg=col,
                 font=("Courier New", 10, "bold")).pack(side=tk.LEFT, padx=6)

        # confidence bar
        bar_bg = tk.Canvas(self, width=60, height=6,
                           bg=BORDER_COLOR, highlightthickness=0)
        bar_bg.pack(side=tk.LEFT, pady=2)
        bar_bg.create_rectangle(0, 0, int(conf * 60), 6, fill=col, outline="")

        tk.Label(self, text=f"{int(conf*100)}%",
                 bg=BG_CARD, fg=TEXT_MUTED,
                 font=("Courier New", 9)).pack(side=tk.LEFT, padx=4)


# ─────────────────────────────────────────────────────────────
#  MAIN APPLICATION
# ─────────────────────────────────────────────────────────────
class ShapeDetectorApp:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("AI Shape Detector")
        self.root.configure(bg=BG_DARK)
        self.root.geometry("1200x750")
        self.root.resizable(True, True)
        self.root.minsize(900, 600)

        # state
        self.cap          = None
        self.running      = False
        self.drawing      = True
        self.tracker      = HandTracker()
        self.drawer       = AirDrawer()
        self.history      = []
        self.history_idx  = 0
        self.last_shape   = "–"
        self.last_conf    = 0.0
        self.fps          = 0
        self._fps_count   = 0
        self._fps_time    = time.time()
        self._last_pred_time = 0
        self._pred_cooldown  = 1.5   # seconds between auto-predictions

        self._build_ui()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    # ── UI CONSTRUCTION ──────────────────────────────────────

    def _build_ui(self):
        # ── HEADER ──────────────────────────────────────────
        header = tk.Frame(self.root, bg=BG_DARK, height=56)
        header.pack(fill=tk.X, padx=0, pady=0)
        header.pack_propagate(False)

        # logo / title
        tk.Label(header,
                 text="◈  AI SHAPE DETECTOR",
                 bg=BG_DARK, fg=ACCENT_CYAN,
                 font=("Courier New", 18, "bold")).pack(side=tk.LEFT, padx=20, pady=10)

        # subtitle
        tk.Label(header,
                 text="real-time gesture recognition",
                 bg=BG_DARK, fg=TEXT_MUTED,
                 font=("Courier New", 10)).pack(side=tk.LEFT, padx=4, pady=14)

        # fps indicator (right)
        self._fps_label = tk.Label(header,
                                    text="FPS: --",
                                    bg=BG_DARK, fg=TEXT_MUTED,
                                    font=("Courier New", 10))
        self._fps_label.pack(side=tk.RIGHT, padx=20)

        # status dot
        self._status_dot = tk.Canvas(header, width=12, height=12,
                                      bg=BG_DARK, highlightthickness=0)
        self._status_dot.create_oval(2, 2, 10, 10, fill="#444", tags="dot")
        self._status_dot.pack(side=tk.RIGHT, padx=(0, 4))

        # separator
        sep = tk.Frame(self.root, bg=BORDER_COLOR, height=1)
        sep.pack(fill=tk.X)

        # ── BODY ─────────────────────────────────────────────
        body = tk.Frame(self.root, bg=BG_DARK)
        body.pack(fill=tk.BOTH, expand=True, padx=12, pady=12)

        # left panel: webcam
        left = tk.Frame(body, bg=BG_DARK)
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self._cam_label = tk.Label(left, bg=BG_PANEL,
                                    text="▶  press START to begin",
                                    fg=TEXT_MUTED,
                                    font=("Courier New", 13))
        self._cam_label.pack(fill=tk.BOTH, expand=True,
                              padx=(0, 8), pady=(0, 8))

        # canvas overlay label
        self._canvas_label = tk.Label(left, bg=BG_PANEL,
                                       text="drawing canvas",
                                       fg=TEXT_MUTED,
                                       font=("Courier New", 11))
        self._canvas_label.pack(fill=tk.BOTH, expand=True,
                                 padx=(0, 8), pady=(0, 0))

        # right panel
        right = tk.Frame(body, bg=BG_DARK, width=280)
        right.pack(side=tk.RIGHT, fill=tk.Y)
        right.pack_propagate(False)

        # ── RESULT CARD ───────────────────────────────────────
        res_card = tk.Frame(right, bg=BG_PANEL,
                             highlightbackground=BORDER_COLOR,
                             highlightthickness=1)
        res_card.pack(fill=tk.X, pady=(0, 8))

        tk.Label(res_card, text="DETECTED SHAPE",
                 bg=BG_PANEL, fg=TEXT_MUTED,
                 font=("Courier New", 9)).pack(anchor="w", padx=12, pady=(10, 2))

        self._shape_var = tk.StringVar(value="–")
        self._shape_label = tk.Label(res_card,
                                      textvariable=self._shape_var,
                                      bg=BG_PANEL, fg=ACCENT_CYAN,
                                      font=("Courier New", 30, "bold"))
        self._shape_label.pack(pady=(0, 4))

        # confidence ring
        ring_row = tk.Frame(res_card, bg=BG_PANEL)
        ring_row.pack(pady=(0, 12))
        self._ring = ConfidenceRing(ring_row, size=100)
        self._ring.pack(side=tk.LEFT, padx=10)

        ring_info = tk.Frame(ring_row, bg=BG_PANEL)
        ring_info.pack(side=tk.LEFT)
        tk.Label(ring_info, text="confidence",
                 bg=BG_PANEL, fg=TEXT_MUTED,
                 font=("Courier New", 9)).pack(anchor="w")
        self._conf_var = tk.StringVar(value="0%")
        tk.Label(ring_info, textvariable=self._conf_var,
                 bg=BG_PANEL, fg=TEXT_PRIMARY,
                 font=("Courier New", 20, "bold")).pack(anchor="w")

        # ── CONTROLS ─────────────────────────────────────────
        ctrl_card = tk.Frame(right, bg=BG_PANEL,
                              highlightbackground=BORDER_COLOR,
                              highlightthickness=1)
        ctrl_card.pack(fill=tk.X, pady=(0, 8))

        tk.Label(ctrl_card, text="CONTROLS",
                 bg=BG_PANEL, fg=TEXT_MUTED,
                 font=("Courier New", 9)).pack(anchor="w", padx=12, pady=(10, 6))

        btn_cfg = dict(
            font=("Courier New", 10, "bold"),
            relief=tk.FLAT, cursor="hand2",
            padx=10, pady=6, width=22
        )

        self._start_btn = tk.Button(ctrl_card,
                                     text="▶  START CAMERA",
                                     bg=ACCENT_GREEN, fg=BG_DARK,
                                     activebackground="#00CC7A",
                                     command=self._start_cam, **btn_cfg)
        self._start_btn.pack(padx=12, pady=3)

        self._stop_btn = tk.Button(ctrl_card,
                                    text="■  STOP CAMERA",
                                    bg=BORDER_COLOR, fg=TEXT_MUTED,
                                    activebackground=ACCENT_PINK,
                                    command=self._stop_cam,
                                    state=tk.DISABLED, **btn_cfg)
        self._stop_btn.pack(padx=12, pady=3)

        self._toggle_draw_btn = tk.Button(ctrl_card,
                                           text="✎  DRAWING ON",
                                           bg=ACCENT_CYAN, fg=BG_DARK,
                                           activebackground=GLOW_CYAN,
                                           command=self._toggle_drawing, **btn_cfg)
        self._toggle_draw_btn.pack(padx=12, pady=3)

        self._predict_btn = tk.Button(ctrl_card,
                                       text="◈  PREDICT NOW",
                                       bg=ACCENT_PURPLE, fg=BG_DARK,
                                       activebackground="#9B59B6",
                                       command=self._predict_now, **btn_cfg)
        self._predict_btn.pack(padx=12, pady=3)

        self._clear_btn = tk.Button(ctrl_card,
                                     text="⌫  CLEAR CANVAS",
                                     bg="#1C2333", fg=TEXT_PRIMARY,
                                     activebackground=ACCENT_PINK,
                                     command=self._clear_canvas, **btn_cfg)
        self._clear_btn.pack(padx=12, pady=(3, 12))

        # ── SHORTCUTS ─────────────────────────────────────────
        hint_card = tk.Frame(right, bg=BG_PANEL,
                              highlightbackground=BORDER_COLOR,
                              highlightthickness=1)
        hint_card.pack(fill=tk.X, pady=(0, 8))

        tk.Label(hint_card, text="KEYBOARD",
                 bg=BG_PANEL, fg=TEXT_MUTED,
                 font=("Courier New", 9)).pack(anchor="w", padx=12, pady=(10, 4))

        shortcuts = [("S", "start camera"),
                     ("X", "stop camera"),
                     ("D", "toggle drawing"),
                     ("P", "predict"),
                     ("C", "clear canvas"),
                     ("Q", "quit")]
        for key, desc in shortcuts:
            row = tk.Frame(hint_card, bg=BG_PANEL)
            row.pack(fill=tk.X, padx=12, pady=1)
            tk.Label(row, text=f"[{key}]",
                     bg=BG_DARK, fg=ACCENT_CYAN,
                     font=("Courier New", 9, "bold"),
                     width=4, padx=2).pack(side=tk.LEFT)
            tk.Label(row, text=desc,
                     bg=BG_PANEL, fg=TEXT_MUTED,
                     font=("Courier New", 9)).pack(side=tk.LEFT, padx=4)

        tk.Frame(hint_card, bg=BG_PANEL, height=8).pack()

        # ── HISTORY ───────────────────────────────────────────
        hist_header = tk.Frame(right, bg=BG_DARK)
        hist_header.pack(fill=tk.X)

        tk.Label(hist_header, text="DETECTION HISTORY",
                 bg=BG_DARK, fg=TEXT_MUTED,
                 font=("Courier New", 9)).pack(side=tk.LEFT)

        tk.Button(hist_header, text="clear",
                  bg=BG_DARK, fg=TEXT_MUTED,
                  font=("Courier New", 8), relief=tk.FLAT,
                  cursor="hand2",
                  command=self._clear_history).pack(side=tk.RIGHT)

        hist_outer = tk.Frame(right, bg=BG_PANEL,
                               highlightbackground=BORDER_COLOR,
                               highlightthickness=1)
        hist_outer.pack(fill=tk.BOTH, expand=True, pady=(4, 0))

        self._hist_frame = tk.Frame(hist_outer, bg=BG_PANEL)
        self._hist_frame.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

        # ── KEY BINDINGS ─────────────────────────────────────
        self.root.bind("<Key-s>", lambda e: self._start_cam())
        self.root.bind("<Key-x>", lambda e: self._stop_cam())
        self.root.bind("<Key-d>", lambda e: self._toggle_drawing())
        self.root.bind("<Key-p>", lambda e: self._predict_now())
        self.root.bind("<Key-c>", lambda e: self._clear_canvas())
        self.root.bind("<Key-q>", lambda e: self._on_close())

    # ── CAMERA CONTROL ───────────────────────────────────────

    def _start_cam(self):
        if self.running:
            return
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            self._shape_var.set("NO CAM")
            return
        self.running = True
        self._start_btn.config(state=tk.DISABLED, bg=BORDER_COLOR, fg=TEXT_MUTED)
        self._stop_btn.config(state=tk.NORMAL, bg=ACCENT_PINK, fg=BG_DARK)
        self._set_status(True)
        t = threading.Thread(target=self._video_loop, daemon=True)
        t.start()

    def _stop_cam(self):
        self.running = False
        if self.cap:
            self.cap.release()
            self.cap = None
        self._start_btn.config(state=tk.NORMAL, bg=ACCENT_GREEN, fg=BG_DARK)
        self._stop_btn.config(state=tk.DISABLED, bg=BORDER_COLOR, fg=TEXT_MUTED)
        self._set_status(False)
        self._cam_label.config(image="",
                               text="▶  press START to begin",
                               fg=TEXT_MUTED, font=("Courier New", 13))
        self._canvas_label.config(image="",
                                  text="drawing canvas",
                                  fg=TEXT_MUTED, font=("Courier New", 11))

    def _toggle_drawing(self):
        self.drawing = not self.drawing
        if self.drawing:
            self._toggle_draw_btn.config(text="✎  DRAWING ON",
                                          bg=ACCENT_CYAN, fg=BG_DARK)
        else:
            self._toggle_draw_btn.config(text="✎  DRAWING OFF",
                                          bg=BORDER_COLOR, fg=TEXT_MUTED)

    def _clear_canvas(self):
        self.drawer.clear()

    def _predict_now(self):
        if self.drawer.get_point_count() > 30:
            shape, conf = classify_shape(self.drawer.canvas)
            self._update_result(shape, conf)
            self._add_history(shape, conf)

    def _set_status(self, on):
        col = ACCENT_GREEN if on else "#444"
        self._status_dot.itemconfig("dot", fill=col)

    # ── VIDEO LOOP (background thread) ────────────────────────

    def _video_loop(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            frame, landmarks = self.tracker.find_hand(frame)

            # draw finger trail
            self.drawer.update(landmarks, drawing=self.drawing)

            # overlay canvas on frame
            canvas = self.drawer.canvas
            mask = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
            _, mask = cv2.threshold(mask, 10, 255, cv2.THRESH_BINARY)
            frame_h, frame_w = frame.shape[:2]
            canvas_r = cv2.resize(canvas, (frame_w, frame_h))
            mask_r   = cv2.resize(mask, (frame_w, frame_h))
            mask_inv = cv2.bitwise_not(mask_r)
            bg  = cv2.bitwise_and(frame, frame, mask=mask_inv)
            fg  = cv2.bitwise_and(canvas_r, canvas_r, mask=mask_r)
            combined = cv2.add(bg, fg)

            # draw HUD border
            cv2.rectangle(combined, (0, 0),
                          (frame_w-1, frame_h-1),
                          (0, 229, 255), 2)

            # auto-predict
            now = time.time()
            pts = self.drawer.get_point_count()
            if (pts > 80 and
                    now - self._last_pred_time > self._pred_cooldown):
                shape, conf = classify_shape(self.drawer.canvas)
                self._last_pred_time = now
                self.root.after(0, self._update_result, shape, conf)
                self.root.after(0, self._add_history, shape, conf)

            # HUD text
            mode_text = "DRAW" if self.drawing else "VIEW"
            cv2.putText(combined, f"MODE: {mode_text}",
                        (10, 28), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 229, 255), 1)
            cv2.putText(combined, f"PTS: {pts}",
                        (10, 52), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (179, 136, 255), 1)
            cv2.putText(combined, f"FPS: {self.fps}",
                        (frame_w - 90, 28), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 229, 255), 1)

            # FPS
            self._fps_count += 1
            elapsed = time.time() - self._fps_time
            if elapsed >= 1.0:
                self.fps = int(self._fps_count / elapsed)
                self._fps_count = 0
                self._fps_time = time.time()
                self.root.after(0, self._fps_label.config,
                                {"text": f"FPS: {self.fps}"})

            # convert and show
            rgb_combined = cv2.cvtColor(combined, cv2.COLOR_BGR2RGB)
            pil_cam   = Image.fromarray(rgb_combined)
            pil_cam   = pil_cam.resize((580, 310), Image.LANCZOS)
            imgtk_cam = ImageTk.PhotoImage(pil_cam)

            rgb_canvas  = cv2.cvtColor(canvas_r, cv2.COLOR_BGR2RGB)
            pil_canvas  = Image.fromarray(rgb_canvas)
            pil_canvas  = pil_canvas.resize((580, 200), Image.LANCZOS)
            imgtk_canvas = ImageTk.PhotoImage(pil_canvas)

            self.root.after(0, self._update_labels,
                            imgtk_cam, imgtk_canvas)

    def _update_labels(self, imgtk_cam, imgtk_canvas):
        self._cam_label.config(image=imgtk_cam, text="")
        self._cam_label.image = imgtk_cam
        self._canvas_label.config(image=imgtk_canvas, text="")
        self._canvas_label.image = imgtk_canvas

    # ── RESULT UPDATE ─────────────────────────────────────────

    def _update_result(self, shape, conf):
        col = SHAPE_COLORS.get(shape, TEXT_MUTED)
        self._shape_var.set(shape.upper())
        self._shape_label.config(fg=col)
        self._conf_var.set(f"{int(conf*100)}%")
        self._ring.set(conf, col)

    # ── HISTORY ───────────────────────────────────────────────

    def _add_history(self, shape, conf):
        if shape == "unknown":
            return
        self.history_idx += 1
        entry = HistoryEntry(self._hist_frame, shape, conf, self.history_idx)
        entry.pack(fill=tk.X, pady=2, padx=2)
        self.history.append(entry)
        # keep last 8
        if len(self.history) > 8:
            old = self.history.pop(0)
            old.destroy()

    def _clear_history(self):
        for w in self.history:
            w.destroy()
        self.history.clear()
        self.history_idx = 0

    # ── CLOSE ─────────────────────────────────────────────────

    def _on_close(self):
        self.running = False
        if self.cap:
            self.cap.release()
        self.root.destroy()

    def run(self):
        self.root.mainloop()


# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app = ShapeDetectorApp()
    app.run()