import os
import sys
import cv2
import numpy as np
import threading
import time
import io
import datetime
import ctypes
import traceback
import shutil
import subprocess 
import gc 
import queue
import asyncio
import tkinter as tk
import re
from tkinter import font as tkfont

# --- NATIVE WINDOWS OCR & CAMERA IMPORTS ---
try:
    from winsdk.windows.media.ocr import OcrEngine
    from winsdk.windows.globalization import Language
    from winsdk.windows.graphics.imaging import SoftwareBitmap, BitmapPixelFormat, BitmapAlphaMode
    from winsdk.windows.security.cryptography import CryptographicBuffer
    from winsdk.windows.devices.enumeration import DeviceInformation, DeviceClass, Panel
    from winsdk.windows.media.capture import MediaCapture, MediaCaptureInitializationSettings, StreamingCaptureMode, MediaCaptureMemoryPreference
    from winsdk.windows.media.capture.frames import MediaFrameSourceKind
    from winsdk.windows.storage.streams import Buffer
    from winsdk.windows.media.devices import FocusSettings, FocusMode, AutoFocusRange
except ImportError:
    print("WARNING: winsdk not found. Please run: pip install winsdk")

# --- BARCODE IMPORT ---
try:
    import barcode
    from barcode.writer import ImageWriter
except ImportError:
    print("WARNING: python-barcode not found. Please run: pip install python-barcode")

# --- PANASONIC DPI SCALING FIX ---
try:
    ctypes.windll.shcore.SetProcessDpiAwareness(2)
except Exception:
    pass

# --- CONSTANTS & INITIAL FALLBACKS ---
CAMERA_W = 1920
CAMERA_H = 1080

TARGET_FPS = 60
FRAME_BUDGET = 1.0 / TARGET_FPS
DRAG_RADIUS = 75       
BTN_H_MAIN = 150 
BTN_H_SMALL = 150       
LIVE_REDRAW_EVERY = 1  
PAD_DETECT = 5 
# --- ENHANCEMENT: Focus Meter Constants ---
ENABLE_FOCUS_ASSIST = False # <-- TOGGLE: Set to False to disable the focus patch and blur detection
FOCUS_THRESHOLD = 15.0    # Lowered to 15.0 to account for document text
PATCH_W = 800          # The size of the center box to evaluate
PATCH_H = 400
FOCUS_CHECK_EVERY = 1     # Run math every Nth frame to save CPU

if getattr(sys, 'frozen', False):
    application_path = os.path.dirname(sys.executable)
else:
    application_path = os.path.dirname(os.path.abspath(__file__))

GT_DIR = os.path.join(application_path, "GroundTruth_Data")
os.makedirs(os.path.join(GT_DIR, "Detection", "images"), exist_ok=True)
os.makedirs(os.path.join(GT_DIR, "Detection", "labels"), exist_ok=True)
os.makedirs(os.path.join(GT_DIR, "OCR", "images"), exist_ok=True)

# --- GLOBAL STATE CONSOLIDATION ---
_S = {
    "state": "LIVE",
    "frozen_frame": None,
    "detected_boxes": [],
    "session_scans": [],
    "session_images": [],
    "session_polygons": [],
    "mismatch_info": "",
    "barcode_img": None,
    "session_folder": "",
    "scan_index": 1,
    "global_error_message": "",
    "retake_index": -1,
    "active_edit_index": -1,
    "roi_start": None,
    "roi_end": None,
    "roi_drawing": False,
    "selected_polygon": None,
    "dragging_point_idx": -1,
    "hover_point_idx": -1,
    "zoom_params": {},
    "extracted_text": "",
    "debug_warped": None,
    "dirty": True,           
    "session_start_time": None,
    "session_duration": 0,
    "focus_score": 0.0,                    # Tracks real-time score
    "focus_color": (0, 0, 255),            # Default Red
    "focus_text": "CALCULATING FOCUS..."   # Default Text
}

# --- CACHES & THREAD QUEUES ---
_text_size_cache = {}
_cmd_q = queue.SimpleQueue()
_ocr_engine_lock = threading.Lock()
_global_ocr_engine = None

# --- DEDICATED OCR EVENT LOOP ---
# Prevents memory leaks by reusing a single event loop for all COM/WinRT OCR tasks
_ocr_loop = asyncio.new_event_loop()
def _start_ocr_loop():
    asyncio.set_event_loop(_ocr_loop)
    _ocr_loop.run_forever()

threading.Thread(target=_start_ocr_loop, daemon=True).start()

# --- INITIAL COMPOSITING CANVAS ALLOCATION ---
display = np.zeros((CAMERA_H, CAMERA_W, 3), dtype=np.uint8)
active_buttons = []
text_edit_zones = []

def _get_ocr_engine():
    global _global_ocr_engine
    if _global_ocr_engine is None:
        with _ocr_engine_lock:
            if _global_ocr_engine is None:
                _global_ocr_engine = OcrEngine.try_create_from_user_profile_languages()
                if _global_ocr_engine is None:
                    print("ERROR: No OCR Language Packs installed on Windows.")
    return _global_ocr_engine

def _get_cached_text_size(text, font, font_scale, thickness):
    key = (text, font, font_scale, thickness)
    if key not in _text_size_cache:
        _text_size_cache[key] = cv2.getTextSize(text, font, font_scale, thickness)[0]
    return _text_size_cache[key]

def show_osk():
    try:
        subprocess.Popen(['C:\\Program Files\\Common Files\\microsoft shared\\ink\\TabTip.exe'], shell=True)
    except Exception as e:
        print(f"Failed to open OSK: {e}")

def show_tablet_edit_dialog(current_text):
    root = tk.Tk()
    root.withdraw()  
    
    dialog = tk.Toplevel(root)
    dialog.title("Edit Scan Entry")
    dialog.attributes("-topmost", True)
    dialog.geometry("850x260+535+410")
    dialog.configure(bg="#1e1e1e")
    dialog.resizable(False, False)
    
    custom_font = tkfont.Font(family="Segoe UI", size=20)
    
    lbl = tk.Label(dialog, text="Tap anywhere inside the text to position cursor:", fg="#00ff00", bg="#1e1e1e", font=("Segoe UI", 14, "bold"))
    lbl.pack(pady=15)
    
    entry_var = tk.StringVar(value=current_text)
    entry = tk.Entry(dialog, textvariable=entry_var, font=custom_font, width=45, bg="#333333", fg="white", insertbackground="white", bd=3, relief=tk.FLAT)
    entry.pack(pady=10, padx=25)
    
    entry.focus_set()
    entry.icursor(tk.END)
    
    output_container = [current_text]
    
    def on_confirm():
        output_container[0] = entry_var.get()
        root.destroy()
        
    def on_cancel():
        root.destroy()

    btn_frame = tk.Frame(dialog, bg="#1e1e1e")
    btn_frame.pack(pady=20)
    
    ok_btn = tk.Button(btn_frame, text="  SAVE CHANGES  ", font=("Segoe UI", 14, "bold"), bg="#00aa00", fg="white", activebackground="#008800", activeforeground="white", command=on_confirm, relief=tk.FLAT, padx=15, pady=8)
    ok_btn.pack(side=tk.LEFT, padx=25)
    
    cancel_btn = tk.Button(btn_frame, text="  DISCARD  ", font=("Segoe UI", 14), bg="#aa0000", fg="white", activebackground="#880000", activeforeground="white", command=on_cancel, relief=tk.FLAT, padx=15, pady=8)
    cancel_btn.pack(side=tk.LEFT, padx=25)
    
    show_osk()
    dialog.mainloop()
    return output_container[0]

def save_ground_truth():
    if not _S["session_scans"]: return
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    for i, text in enumerate(_S["session_scans"]):
        if "ERROR" in text or text == "[NO TEXT FOUND]": continue
            
        if i < len(_S["session_images"]) and _S["session_images"][i] is not None:
            ocr_filename = f"ocr_{timestamp}_{i}.jpg"
            cv2.imwrite(os.path.join(GT_DIR, "OCR", "images", ocr_filename), _S["session_images"][i])
            with open(os.path.join(GT_DIR, "OCR", "labels.txt"), "a", encoding="utf-8") as f:
                f.write(f"{ocr_filename}\t{text}\n")
            
        raw_cap_path = os.path.join(_S["session_folder"], f"scan_{i+1}_01_RawCapture.jpg")
        if os.path.exists(raw_cap_path) and i < len(_S["session_polygons"]):
            det_filename = f"det_{timestamp}_{i}.jpg"
            shutil.copy(raw_cap_path, os.path.join(GT_DIR, "Detection", "images", det_filename))
            
            poly = _S["session_polygons"][i]
            pts = [str(int(val)) for point in poly for val in point]
            line = ",".join(pts) + f',"{text}"\n'
            
            with open(os.path.join(GT_DIR, "Detection", "labels", f"det_{timestamp}_{i}.txt"), "w", encoding="utf-8") as f:
                f.write(line)

def cleanup_and_exit():
    global cam
    if 'cam' in globals() and cam is not None:
        cam.stop()
    cv2.destroyAllWindows()
    gc.collect()
    os._exit(0)

def init_new_session():
    _S["session_scans"].clear()
    _S["session_images"].clear()
    _S["session_polygons"].clear()
    _S["active_edit_index"] = -1
    _S["retake_index"] = -1
    _S["barcode_img"] = None
    _S["state"] = "LIVE"
    _S["roi_start"] = _S["roi_end"] = None
    _S["scan_index"] = 1
    _S["session_start_time"] = time.time()
    _S["session_duration"] = 0
    _S["dirty"] = True
    _S["focus_score"] = 0.0
    _S["focus_color"] = (0, 0, 255)
    _S["focus_text"] = "CALCULATING FOCUS..."
        
    base_dir = os.path.join(application_path, "SmartScanner_Sessions_DB")
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    _S["session_folder"] = os.path.join(base_dir, f"session_{timestamp}")
    os.makedirs(_S["session_folder"], exist_ok=True)

init_new_session()

class WinsdkCamera:
    def __init__(self, width=1920, height=1080):
        self.width = width
        self.height = height
        self.frame = np.zeros((height, width, 3), dtype=np.uint8)
        self.ret = False
        self.stopped = False
        self.media_capture = None
        self.frame_reader = None
        self.torch_control = None
        self.flash_on = False
        
        self.loop = asyncio.new_event_loop()
        threading.Thread(target=self._start_loop, daemon=True).start()

    def _start_loop(self):
        asyncio.set_event_loop(self.loop)
        self.loop.run_until_complete(self._capture_loop())

    async def _capture_loop(self):
        devices = await DeviceInformation.find_all_async(DeviceClass.VIDEO_CAPTURE)
        rear_device_id = None
        for d in devices:
            if d.enclosure_location and d.enclosure_location.panel == Panel.BACK:
                rear_device_id = d.id
                break
        if not rear_device_id and devices.size > 0:
            rear_device_id = devices[0].id
            
        self.media_capture = MediaCapture()
        settings = MediaCaptureInitializationSettings()
        if rear_device_id:
            settings.video_device_id = rear_device_id
        settings.streaming_capture_mode = StreamingCaptureMode.VIDEO
        settings.memory_preference = MediaCaptureMemoryPreference.CPU
        await self.media_capture.initialize_async(settings)
        
        controller = self.media_capture.video_device_controller
        if controller.focus_control.supported:
            focus_settings = FocusSettings()
            focus_settings.mode = FocusMode.CONTINUOUS
            focus_settings.auto_focus_range = AutoFocusRange.FULL_RANGE
            controller.focus_control.configure(focus_settings)
        self.torch_control = controller.torch_control
        if self.torch_control.supported:
            self.torch_control.enabled = self.flash_on
            
        color_source = None
        for source in self.media_capture.frame_sources.values():
            if source.info.source_kind == MediaFrameSourceKind.COLOR:
                color_source = source
                break
        if not color_source: return
            
        self.frame_reader = await self.media_capture.create_frame_reader_async(color_source)
        await self.frame_reader.start_async()
        
        while not self.stopped:
            try:
                frame_reference = self.frame_reader.try_acquire_latest_frame()
                if frame_reference and frame_reference.video_media_frame:
                    bm = frame_reference.video_media_frame.software_bitmap
                    if bm:
                        if bm.bitmap_pixel_format != BitmapPixelFormat.BGRA8:
                            bm = SoftwareBitmap.convert(bm, BitmapPixelFormat.BGRA8)
                        buf = Buffer(bm.pixel_width * bm.pixel_height * 4)
                        bm.copy_to_buffer(buf)
                        byte_data = CryptographicBuffer.copy_to_byte_array(buf)
                        if isinstance(byte_data, tuple):
                            byte_data = bytes(byte_data)
                            
                        img_np = np.frombuffer(byte_data, dtype=np.uint8).reshape((bm.pixel_height, bm.pixel_width, 4))
                        self.frame = cv2.cvtColor(img_np, cv2.COLOR_BGRA2BGR)
                        self.ret = True
            except Exception:
                pass
            await asyncio.sleep(0.005)

    def read(self):
        return self.ret, self.frame

    def snapshot(self):
        return self.frame.copy() if self.ret else None

    def toggle_flash(self):
        if self.torch_control and self.torch_control.supported:
            self.flash_on = not self.flash_on
            self.torch_control.enabled = self.flash_on
            return self.flash_on
        return False

    def stop(self):
        self.stopped = True
        if self.torch_control and self.torch_control.supported:
            self.torch_control.enabled = False
        if self.frame_reader: self.frame_reader.close()
        if self.media_capture: self.media_capture.close()

def _cv_to_software_bitmap(img):
    if len(img.shape) == 3 and img.shape[2] == 3: 
        img_bgra = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    elif len(img.shape) == 2: 
        img_bgra = cv2.cvtColor(img, cv2.COLOR_GRAY2BGRA)
    else: 
        img_bgra = img
        
    h, w, _ = img_bgra.shape
    ibuffer = CryptographicBuffer.create_from_byte_array(img_bgra.tobytes())
    return SoftwareBitmap.create_copy_from_buffer(
        ibuffer, BitmapPixelFormat.BGRA8, w, h, BitmapAlphaMode.PREMULTIPLIED
    )

async def windows_native_detect(cv_img):
    engine = _get_ocr_engine()
    if engine is None: raise Exception("OCR Engine initialization failed.")
    
    software_bitmap = _cv_to_software_bitmap(cv_img)
    result = await engine.recognize_async(software_bitmap)
    
    boxes = []
    for line in result.lines:
        if not line.words: continue
        xmin, ymin, xmax, ymax = float('inf'), float('inf'), 0, 0
        for word in line.words:
            rect = word.bounding_rect
            if rect.x < xmin: xmin = rect.x
            if rect.y < ymin: ymin = rect.y
            if (rect.x + rect.width) > xmax: xmax = rect.x + rect.width
            if (rect.y + rect.height) > ymax: ymax = rect.y + rect.height
        boxes.append([xmin, xmax, ymin, ymax])
    return boxes

async def windows_native_ocr(cv_img):
    engine = _get_ocr_engine()
    if engine is None: return "ERROR: No OCR Engine Available"
    
    software_bitmap = _cv_to_software_bitmap(cv_img)
    result = await engine.recognize_async(software_bitmap)
    return result.text

def run_detection_thread(img, rx1, ry1, rx2, ry2):
    try:
        crop = img[ry1:ry2, rx1:rx2]
        # Grab the absolute image dimensions for safe clamping
        img_h, img_w = img.shape[:2] 
        
        # Use threadsafe submission to dedicated OCR loop
        future = asyncio.run_coroutine_threadsafe(windows_native_detect(crop), _ocr_loop)
        found_boxes = future.result() 
        
        # --- DEFINE PADDING HERE ---
        # 10 to 15 pixels is usually perfect for natural camera images
        
        
        adjusted = []
        for box in found_boxes:
            xmin, xmax, ymin, ymax = box
            
            # Apply padding and clamp it safely inside the image boundaries
            pad_xmin = max(0, (xmin + rx1) - PAD_DETECT)
            pad_xmax = min(img_w, (xmax + rx1) + PAD_DETECT)
            pad_ymin = max(0, (ymin + ry1) - PAD_DETECT)
            pad_ymax = min(img_h, (ymax + ry1) + PAD_DETECT)
            
            adjusted.append([pad_xmin, pad_xmax, pad_ymin, pad_ymax])
            
        _cmd_q.put(("SET_DETECTED_BOXES", adjusted))
    except Exception as e:
        _cmd_q.put(("SET_ERROR", f"Detection Error: {str(e)}"))

def run_ocr_thread(img, polygon_pts):
    try:
        warped = four_point_transform(img, polygon_pts)
        if warped.size == 0 or warped.shape[0] < 5 or warped.shape[1] < 5:
            err_img = np.zeros((80, 300, 3), dtype=np.uint8)
            cv2.putText(err_img, "ERR", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
            _cmd_q.put(("OCR_COMPLETED", ("ERROR: Invalid Polygon", err_img, polygon_pts)))
            return

        gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
        # 2. Conditionally upscale ONLY if the text block is relatively small
        height, width = gray.shape
        if height < 800 and width < 800:
            processed_img = cv2.resize(gray, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
        else:
            processed_img = gray # It's already big enough from the tablet camera
        
        
        #padded = cv2.copyMakeBorder(processed_img, 15, 15, 15, 15, cv2.BORDER_CONSTANT, value=255)
        padded = cv2.copyMakeBorder(processed_img, 15, 15, 15, 15, cv2.BORDER_REPLICATE)
        # Use threadsafe submission to dedicated OCR loop
        future = asyncio.run_coroutine_threadsafe(windows_native_ocr(padded), _ocr_loop)
        raw_text = future.result()
        
        #cleaned_text = raw_text.replace('\n', ' ').replace('\r', '').strip()
        
        # --- Regex Update: Permit unicode to keep localized chars ---
        #cleaned_text = re.sub(r'[^\w\s\-\.,/]', '', cleaned_text, flags=re.UNICODE).strip()
        
        if not raw_text: raw_text = "[NO TEXT FOUND]"
        
        display_ready_img = cv2.cvtColor(padded, cv2.COLOR_GRAY2BGR)
        
        _cmd_q.put(("OCR_COMPLETED", (raw_text, display_ready_img, polygon_pts)))
    except Exception as e:
        _cmd_q.put(("SET_ERROR", f"OCR Engine Error: {str(e)}"))

def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def four_point_transform(image, pts):
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    dst = np.array([[0, 0], [maxWidth - 1, 0], [maxWidth - 1, maxHeight - 1], [0, maxHeight - 1]], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(image, M, (maxWidth, maxHeight))

def generate_barcode_cv2(text):
    # Pass options to generate a high-quality, high-contrast base image
    options = {
        "write_text": True,
        "dpi": 300,
        "module_width": 0.3,  # Makes the thinnest bars slightly thicker
        "module_height": 15.0 # Gives the bars decent vertical height
    }
    
    try:
        rv = io.BytesIO()
        code128 = barcode.get('code128', text, writer=ImageWriter())
        code128.write(rv, options=options)
        rv.seek(0)
        img_arr = np.frombuffer(rv.read(), dtype=np.uint8)
        return cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
        
    except Exception as e:
        print(f"Barcode Gen Error: {e}")
        err_img = np.zeros((200, 600, 3), dtype=np.uint8)
        err_img[:] = (0, 0, 255)
        cv2.putText(err_img, "BARCODE GEN ERROR", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        return err_img

def draw_button(img, text, rect, bg_color, text_color=(255, 255, 255)):
    x, y, w, h = rect
    cv2.rectangle(img, (x, y), (x + w, y + h), bg_color, -1)
    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 255), 2) 
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.0
    thickness = 2
    
    text_size = _get_cached_text_size(text, font, font_scale, thickness)
    txt_x = x + (w - text_size[0]) // 2
    txt_y = y + (h + text_size[1]) // 2
    cv2.putText(img, text, (txt_x, txt_y), font, font_scale, text_color, thickness)

def mouse_handler(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        for btn in active_buttons:
            bx, by, bw, bh = btn["rect"]
            if bx <= x <= bx + bw and by <= y <= by + bh:
                _cmd_q.put((btn["cmd"], None))
                return 
                
        if _S["state"] == "RESULT":
            for zone in text_edit_zones:
                zx, zy, zw, zh, idx = zone
                if zx <= x <= zx + zw and zy <= y <= zy + zh:
                    _cmd_q.put(("START_EDIT", idx))
                    return

    if _S["state"] == "SELECT_ROI":
        if event == cv2.EVENT_LBUTTONDOWN:
            _S["roi_start"] = _S["roi_end"] = (x, y)
            _S["roi_drawing"] = True
            _S["dirty"] = True
        elif event == cv2.EVENT_MOUSEMOVE and _S["roi_drawing"]:
            _S["roi_end"] = (x, y)
            _S["dirty"] = True
        elif event == cv2.EVENT_LBUTTONUP:
            _S["roi_drawing"] = False
            _S["dirty"] = True

    elif _S["state"] == "SELECT_BOX":
        if event == cv2.EVENT_LBUTTONDOWN:
            for box in _S["detected_boxes"]:
                xmin, xmax, ymin, ymax = box
                if xmin - 20 <= x <= xmax + 20 and ymin - 20 <= y <= ymax + 20:
                    _S["selected_polygon"] = np.array([
                        [xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]
                    ], dtype="float32")
                    _cmd_q.put(("INIT_ADJUST_POLY", None))
                    break

    elif _S["state"] == "ADJUST_POLY":
        zp = _S["zoom_params"]
        if not zp: return
        ix = (x - zp["off_x"]) / zp["scale"] + zp["zx1"]
        iy = (y - zp["off_y"]) / zp["scale"] + zp["zy1"]
        
        if event == cv2.EVENT_MOUSEMOVE and _S["dragging_point_idx"] != -1:
            _S["selected_polygon"][_S["dragging_point_idx"]] = [ix, iy]
            _S["dirty"] = True
            return

        old_hover = _S["hover_point_idx"]
        _S["hover_point_idx"] = -1
        if _S["selected_polygon"] is not None:
            for i, pt in enumerate(_S["selected_polygon"]):
                sx = (pt[0] - zp["zx1"]) * zp["scale"] + zp["off_x"]
                sy = (pt[1] - zp["zy1"]) * zp["scale"] + zp["off_y"]
                if np.sqrt((sx - x) ** 2 + (sy - y) ** 2) < DRAG_RADIUS:
                    _S["hover_point_idx"] = i
                    break
        if old_hover != _S["hover_point_idx"]: _S["dirty"] = True

        if event == cv2.EVENT_LBUTTONDOWN and _S["hover_point_idx"] != -1:
            _S["dragging_point_idx"] = _S["hover_point_idx"]
            _S["dirty"] = True
        elif event == cv2.EVENT_LBUTTONUP:
            _S["dragging_point_idx"] = -1
            _S["dirty"] = True

def _draw_live(canvas, live_frame):
    np.copyto(canvas, live_frame)
    scan_count = len(_S["session_scans"])
    
    prompt = "Aim Camera at Document" if scan_count == 0 else f"Scan #{scan_count + 1} - Aim Camera"
    h_color = (0, 255, 0) if _S["state"] == "LIVE" else (0, 165, 255)
    if _S["state"] == "LIVE_RETAKE": 
        prompt = f"Retaking Scan #{_S['retake_index'] + 1} - Aim Camera"
    
    cv2.putText(canvas, prompt, (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, h_color, 3)
    
    # --- ENHANCEMENT: Draw Focus Meter UI ---
    if ENABLE_FOCUS_ASSIST:
        fh, fw = canvas.shape[:2]
        px1, py1 = (fw - PATCH_W) // 2, (fh - PATCH_H) // 2
        px2, py2 = px1 + PATCH_W, py1 + PATCH_H
        
        focus_color = _S.get("focus_color", (0, 0, 255))
        focus_text = _S.get("focus_text", "CALCULATING FOCUS...")
        
        cv2.rectangle(canvas, (px1, py1), (px2, py2), focus_color, 4)
        cv2.putText(canvas, focus_text, (px1, py1 - 15), cv2.FONT_HERSHEY_SIMPLEX, 1.0, focus_color, 3)
    # ----------------------------------------
    
    btn_w = 320
    btn_x = CAMERA_W - btn_w - 50                
    btn_y_center = (CAMERA_H // 2) - (BTN_H_MAIN // 2) 
    
    flash_y = btn_y_center - BTN_H_MAIN - 20
    flash_label = "FLASH: ON" if cam.flash_on else "FLASH: OFF"
    flash_color = (0, 200, 200) if cam.flash_on else (100, 100, 100)
    
    active_buttons.append({"rect": (btn_x, flash_y, btn_w, BTN_H_MAIN), "cmd": "CMD_TOGGLE_FLASH", "label": flash_label, "color": flash_color})
    active_buttons.append({"rect": (btn_x, btn_y_center, btn_w, BTN_H_MAIN), "cmd": "CMD_ACTION_1", "label": "CAPTURE", "color": h_color})
    
    if _S["state"] == "LIVE_RETAKE":
        btn_w_cancel = 220
        btn_x_cancel = CAMERA_W - btn_w_cancel - 50
        active_buttons.append({"rect": (btn_x_cancel, CAMERA_H - 140, btn_w_cancel, BTN_H_MAIN), "cmd": "CMD_CANCEL", "label": "CANCEL", "color": (0, 0, 200)})
def _draw_select_roi(canvas):
    np.copyto(canvas, _S["frozen_frame"])
    overlay = canvas.copy()
    cv2.rectangle(overlay, (0, 0), (CAMERA_W, CAMERA_H), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.4, canvas, 0.6, 0, dst=canvas)

    if _S["roi_start"] and _S["roi_end"]:
        rx1, ry1 = max(0, min(_S["roi_start"][0], _S["roi_end"][0])), max(0, min(_S["roi_start"][1], _S["roi_end"][1]))
        rx2, ry2 = min(CAMERA_W, max(_S["roi_start"][0], _S["roi_end"][0])), min(CAMERA_H, max(_S["roi_start"][1], _S["roi_end"][1]))
        if rx2 > rx1 and ry2 > ry1:
            canvas[ry1:ry2, rx1:rx2] = _S["frozen_frame"][ry1:ry2, rx1:rx2]
            cv2.rectangle(canvas, (rx1, ry1), (rx2, ry2), (0, 255, 255), 3)
            
    cv2.putText(canvas, "Drag across text, then tap CONFIRM", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
    
    # --- CONFIRM ROI BUTTON (Right side, vertically centered like CAPTURE) ---
    btn_w_confirm = 320
    btn_x_confirm = CAMERA_W - btn_w_confirm - 50
    btn_y_center = (CAMERA_H // 2) - (BTN_H_MAIN // 2)
    active_buttons.append({"rect": (btn_x_confirm, btn_y_center, btn_w_confirm, BTN_H_MAIN), "cmd": "CMD_ACTION_2", "label": "CONFIRM ROI", "color": (0, 200, 0)})

    # --- CANCEL BUTTON (Bottom right corner) ---
    btn_w_cancel = 220
    btn_x_cancel = CAMERA_W - btn_w_cancel - 50
    btn_y_cancel = CAMERA_H - 140
    active_buttons.append({"rect": (btn_x_cancel, btn_y_cancel, btn_w_cancel, BTN_H_MAIN), "cmd": "CMD_CANCEL", "label": "CANCEL", "color": (0, 0, 200)})


def _draw_detecting(canvas):
    canvas.fill(0)
    dots = "." * (int(time.time() * 3) % 4)
    cv2.putText(canvas, f"DETECTING TEXT{dots}", (CAMERA_W//2 - 200, CAMERA_H//2), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 4)

def _draw_select_box(canvas):
    np.copyto(canvas, _S["frozen_frame"])
    if _S["roi_start"] and _S["roi_end"]:
        rx1, ry1 = max(0, min(_S["roi_start"][0], _S["roi_end"][0])), max(0, min(_S["roi_start"][1], _S["roi_end"][1]))
        rx2, ry2 = min(CAMERA_W, max(_S["roi_start"][0], _S["roi_end"][0])), min(CAMERA_H, max(_S["roi_start"][1], _S["roi_end"][1]))
        cv2.rectangle(canvas, (rx1, ry1), (rx2, ry2), (0, 165, 255), 2)
        
    for box in _S["detected_boxes"]:
        xmin, xmax, ymin, ymax = [int(v) for v in box]
        cv2.rectangle(canvas, (xmin, ymin), (xmax, ymax), (0, 255, 0), 3)
        
    cv2.putText(canvas, "Tap a green box to select it", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
    
    btn_w_cancel = 220
    btn_x_cancel = CAMERA_W - btn_w_cancel - 50
    active_buttons.append({"rect": (btn_x_cancel, CAMERA_H - 140, btn_w_cancel, BTN_H_MAIN), "cmd": "CMD_CANCEL", "label": "BACK", "color": (0, 0, 200)})

def _draw_adjust_poly(canvas):
    canvas.fill(0)
    zp = _S["zoom_params"]
    if not zp: return
    
    crop = _S["frozen_frame"][zp["zy1"]:zp["zy2"], zp["zx1"]:zp["zx2"]]
    scaled_crop = cv2.resize(crop, None, fx=zp["scale"], fy=zp["scale"])
    canvas[zp["off_y"]:zp["off_y"]+scaled_crop.shape[0], zp["off_x"]:zp["off_x"]+scaled_crop.shape[1]] = scaled_crop
    
    pts_screen = []
    for pt in _S["selected_polygon"]:
        sx = int((pt[0] - zp["zx1"]) * zp["scale"] + zp["off_x"])
        sy = int((pt[1] - zp["zy1"]) * zp["scale"] + zp["off_y"])
        pts_screen.append([sx, sy])
        
    pts_screen = np.array(pts_screen)
    cv2.polylines(canvas, [pts_screen], isClosed=True, color=(255, 0, 0), thickness=3)
    
    for i, pt in enumerate(pts_screen):
        color = (0, 0, 255) if i == _S["hover_point_idx"] else (0, 255, 0)
        cv2.circle(canvas, tuple(pt), 16, color, -1) 
        cv2.circle(canvas, tuple(pt), 18, (255,255,255), 2) 
        
    if _S["dragging_point_idx"] != -1:
        ix, iy = int(_S["selected_polygon"][_S["dragging_point_idx"]][0]), int(_S["selected_polygon"][_S["dragging_point_idx"]][1])
        patch_size = 100
        half = patch_size // 2
        padded = cv2.copyMakeBorder(_S["frozen_frame"], half, half, half, half, cv2.BORDER_REPLICATE)
        px, py = ix + half, iy + half
        patch = padded[py - half : py + half, px - half : px + half]
        
        magnified = cv2.resize(patch, None, fx=3, fy=3, interpolation=cv2.INTER_NEAREST)
        mh, mw = magnified.shape[:2]
        cv2.line(magnified, (mw//2, 0), (mw//2, mh), (0, 255, 0), 2)
        cv2.line(magnified, (0, mh//2), (mw, mh//2), (0, 255, 0), 2)
        
        tr_x, tr_y = CAMERA_W - mw - 40, 40
        canvas[tr_y : tr_y + mh, tr_x : tr_x + mw] = magnified
        cv2.rectangle(canvas, (tr_x, tr_y), (tr_x + mw, tr_y + mh), (255, 255, 255), 4)

    cv2.putText(canvas, "Drag corners to adjust, then tap CONFIRM", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 3)
    
    # --- CONFIRM BUTTON (Right side, vertically centered) ---
    btn_w_confirm = 320
    btn_x_confirm = CAMERA_W - btn_w_confirm - 50
    btn_y_center = (CAMERA_H // 2) - (BTN_H_MAIN // 2)
    active_buttons.append({"rect": (btn_x_confirm, btn_y_center, btn_w_confirm, BTN_H_MAIN), "cmd": "CMD_ACTION_2", "label": "CONFIRM", "color": (0, 200, 0)})

    # --- BACK BUTTON (Bottom right corner) ---
    btn_w_cancel = 220
    btn_x_cancel = CAMERA_W - btn_w_cancel - 50
    active_buttons.append({"rect": (btn_x_cancel, CAMERA_H - 140, btn_w_cancel, BTN_H_MAIN), "cmd": "CMD_CANCEL", "label": "BACK", "color": (0, 0, 200)})
    
def _draw_processing(canvas):
    canvas.fill(0)
    dots = "." * (int(time.time() * 4) % 4)
    cv2.putText(canvas, f"READING TEXT{dots}", (CAMERA_W//2 - 200, CAMERA_H//2), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 4)

def _draw_error(canvas):
    cv2.rectangle(canvas, (0, 0), (CAMERA_W, CAMERA_H), (0, 0, 150), -1)
    cv2.putText(canvas, "ATTENTION REQUIRED", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 4)
    cv2.putText(canvas, _S["global_error_message"], (50, 250), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
    active_buttons.append({"rect": (CAMERA_W//2 - 200, CAMERA_H - 140, 400, BTN_H_MAIN), "cmd": "CMD_CANCEL", "label": "DISMISS / RETRY", "color": (100, 100, 100)})

def _draw_result(canvas):
    cv2.rectangle(canvas, (0, 0), (CAMERA_W, CAMERA_H), (30, 30, 30), -1)
    
    start_time = _S["session_start_time"] if _S["session_start_time"] is not None else time.time()
    cur_dur = round(time.time() - start_time, 1)
    
    title = f"SESSION SCANS (Total: {len(_S['session_scans'])} | Time: {cur_dur}s) - Click box to edit"
    cv2.putText(canvas, title, (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    
    start_y = 120
    row_height = 110
    for i, text in enumerate(_S["session_scans"]):
        box_y = start_y + (i * row_height)
        
        if i < len(_S["session_images"]) and _S["session_images"][i] is not None and _S["session_images"][i].size > 0:
            img = _S["session_images"][i]
            ih, iw = img.shape[:2]
            scale = min(90 / ih, 300 / iw)
            nw, nh = int(iw * scale), int(ih * scale)
            thumb = cv2.resize(img, (nw, nh))
            y_off = box_y + (90 - nh) // 2
            canvas[y_off:y_off+nh, 20:20+nw] = thumb
            cv2.rectangle(canvas, (20, y_off), (20+nw, y_off+nh), (200, 200, 200), 1)

        text_x = 340
        text_w = (CAMERA_W - 250) - text_x
        bg_color = (60, 60, 60)
        b_color = (255, 255, 255)
        
        cv2.rectangle(canvas, (text_x, box_y), (text_x + text_w, box_y + 90), bg_color, -1)
        cv2.rectangle(canvas, (text_x, box_y), (text_x + text_w, box_y + 90), b_color, 2)
        
        disp_txt = f"Scan {i+1}: {text}"
        cv2.putText(canvas, disp_txt, (text_x + 15, box_y + 55), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        text_edit_zones.append((text_x, box_y, text_w, 90, i))
        active_buttons.append({"rect": (CAMERA_W - 230, box_y + 15, 180, 60), "cmd": f"CMD_RETAKE_{i}", "label": "RETAKE", "color": (0, 165, 255)})

    btn_w = 280
    gap = 50
    n_btn = 3 if _S["session_scans"] else 2
    start_x = (CAMERA_W - (n_btn * btn_w + (n_btn - 1) * gap)) // 2
    
    active_buttons.append({"rect": (start_x, CAMERA_H - 140, btn_w, BTN_H_MAIN), "cmd": "CMD_ADD_SCAN", "label": "+ ADD SCAN", "color": (200, 100, 0)})
    active_buttons.append({"rect": (start_x + btn_w + gap, CAMERA_H - 140, btn_w, BTN_H_MAIN), "cmd": "CMD_UNDO", "label": "UNDO LAST", "color": (0, 0, 200)})
    if _S["session_scans"]:
         active_buttons.append({"rect": (start_x + 2*(btn_w + gap), CAMERA_H - 140, btn_w, BTN_H_MAIN), "cmd": "CMD_VALIDATE", "label": "VALIDATE & SAVE", "color": (0, 200, 0)})

def _draw_validation_failed(canvas):
    canvas.fill(0)
    cv2.putText(canvas, "VALIDATION FAILED!", (CAMERA_W//2 - 300, CAMERA_H//2 - 100), cv2.FONT_HERSHEY_SIMPLEX, 1.8, (0, 0, 255), 5)
    cv2.putText(canvas, _S["mismatch_info"], (CAMERA_W//2 - 450, CAMERA_H//2), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
    active_buttons.append({"rect": (CAMERA_W//2 - 320, CAMERA_H - 140, 300, BTN_H_MAIN), "cmd": "CMD_UNDO", "label": "UNDO MISTAKE", "color": (0, 0, 200)})
    active_buttons.append({"rect": (CAMERA_W//2 + 20, CAMERA_H - 140, 300, BTN_H_MAIN), "cmd": "CMD_NEW_SESSION", "label": "DISCARD ALL", "color": (100, 100, 100)})

def _draw_barcode_result(canvas):
    canvas.fill(0)
    cv2.putText(canvas, f"VALIDATION SUCCESSFUL! (Time: {_S['session_duration']}s)", (CAMERA_W//2 - 450, 120), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 4)
    
    # --- DISCLAIMER FOR LONG TEXT ---
    if _S["session_scans"] and len(_S["session_scans"][0]) > 30:
        warn_text = f"WARNING: Text is {len(_S['session_scans'][0])} chars. Barcode might be too dense to scan."
        # Use your existing text cache to perfectly center the warning
        tw, th = _get_cached_text_size(warn_text, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 3)
        cv2.putText(canvas, warn_text, ((CAMERA_W - tw) // 2, 180), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 165, 255), 3)
    
    if _S["barcode_img"] is not None:
        bh, bw = _S["barcode_img"].shape[:2]
        
        # --- AUTO-SCALING LOGIC ---
        # Set max boundaries: 85% of screen width, 60% of screen height
        max_w = int(CAMERA_W * 0.5)
        max_h = int(CAMERA_H * 0.5)
        
        # Calculate scale factor to perfectly fit those boundaries
        scale = min(max_w / bw, max_h / bh)
        
        new_w = int(bw * scale)
        new_h = int(bh * scale)
        
        # CRITICAL: Use INTER_NEAREST for barcodes. 
        # Standard interpolation blurs the black/white edges, making them unscannable.
        scaled_barcode = cv2.resize(_S["barcode_img"], (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        
        # Center the newly scaled barcode on the canvas
        bx, by = (CAMERA_W - new_w) // 2, (CAMERA_H - new_h) // 2
        canvas[by:by+new_h, bx:bx+new_w] = scaled_barcode

    active_buttons.append({"rect": (CAMERA_W//2 - 175, CAMERA_H - 140, 350, BTN_H_MAIN), "cmd": "CMD_NEW_SESSION", "label": "NEW SESSION", "color": (0, 200, 0)})
def _handle_cmd(cmd, payload):
    _S["dirty"] = True 
    
    if cmd == "QUIT":
        cleanup_and_exit()
    elif cmd == "CMD_CANCEL":
        if _S["state"] == "LIVE_RETAKE":
            _S["state"] = "RESULT"
            _S["retake_index"] = -1
        elif _S["state"] in ["SELECT_ROI", "SELECT_BOX", "ADJUST_POLY", "ERROR"]:
            _S["state"] = "LIVE_RETAKE" if _S["retake_index"] != -1 else "LIVE"
            _S["roi_start"] = _S["roi_end"] = None
    elif cmd == "CMD_TOGGLE_FLASH":
        cam.toggle_flash()
    elif cmd == "CMD_ACTION_1":
        if _S["state"] in ["LIVE", "LIVE_RETAKE"]:
            snap = cam.snapshot()
            if snap is not None:
                passed_focus_check = True
                
                # --- ENHANCEMENT: Final Blur Validation Check (Center Patch Only) ---
                if ENABLE_FOCUS_ASSIST:
                    fh, fw = snap.shape[:2]
                    px1, py1 = max(0, (fw - PATCH_W) // 2), max(0, (fh - PATCH_H) // 2)
                    patch = snap[py1:py1+PATCH_H, px1:px1+PATCH_W] 
                    
                    gray_patch = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
                    laplacian_var = cv2.Laplacian(gray_patch, cv2.CV_64F).var()
                    
                    if laplacian_var < FOCUS_THRESHOLD:
                        _S["global_error_message"] = f"Image blurry or moving (Score: {laplacian_var:.1f}). Center the text and hold still!"
                        _S["state"] = "ERROR"
                        passed_focus_check = False
                
                if passed_focus_check:
                    _S["frozen_frame"] = snap
                    lbl = _S["retake_index"] + 1 if _S["retake_index"] != -1 else _S["scan_index"]
                    cv2.imwrite(os.path.join(_S["session_folder"], f"scan_{lbl}_01_RawCapture.jpg"), _S["frozen_frame"])
                    _S["state"] = "SELECT_ROI"
                    _S["roi_start"] = _S["roi_end"] = None
                    
    elif cmd == "CMD_ACTION_2":
        if _S["state"] == "SELECT_ROI":
            if _S["roi_start"] and _S["roi_end"] and abs(_S["roi_start"][0] - _S["roi_end"][0]) > 10 and abs(_S["roi_start"][1] - _S["roi_end"][1]) > 10:
                rx1, ry1 = max(0, min(_S["roi_start"][0], _S["roi_end"][0])), max(0, min(_S["roi_start"][1], _S["roi_end"][1]))
                rx2, ry2 = min(CAMERA_W, max(_S["roi_start"][0], _S["roi_end"][0])), min(CAMERA_H, max(_S["roi_start"][1], _S["roi_end"][1]))
                _S["state"] = "DETECTING"
                threading.Thread(target=run_detection_thread, args=(_S["frozen_frame"], rx1, ry1, rx2, ry2), daemon=True).start()
        elif _S["state"] == "ADJUST_POLY":
            _S["state"] = "PROCESSING"
            threading.Thread(target=run_ocr_thread, args=(_S["frozen_frame"], _S["selected_polygon"]), daemon=True).start()
    elif cmd == "SET_DETECTED_BOXES":
        _S["detected_boxes"] = payload
        _S["state"] = "SELECT_BOX"
    elif cmd == "INIT_ADJUST_POLY":
        xmin, xmax, ymin, ymax = _S["detected_boxes"][0] if not len(_S["detected_boxes"]) == 0 else (0,0,0,0)
        if _S["selected_polygon"] is not None:
            xs, ys = _S["selected_polygon"][:, 0], _S["selected_polygon"][:, 1]
            xmin, xmax, ymin, ymax = min(xs), max(xs), min(ys), max(ys)
        padding = 100
        zx1, zy1 = max(0, int(xmin) - padding), max(0, int(ymin) - padding)
        zx2, zy2 = min(CAMERA_W, int(xmax) + padding), min(CAMERA_H, int(ymax) + padding)
        scale = min(CAMERA_W / (zx2 - zx1), CAMERA_H / (zy2 - zy1))
        _S["zoom_params"] = {
            "zx1": zx1, "zy1": zy1, "zx2": zx2, "zy2": zy2,
            "scale": scale, "off_x": (CAMERA_W - int((zx2 - zx1) * scale)) // 2, "off_y": (CAMERA_H - int((zy2 - zy1) * scale)) // 2
        }
        _S["state"] = "ADJUST_POLY"
    elif cmd == "OCR_COMPLETED":
        text, display_ready_img, poly = payload
        if _S["retake_index"] != -1:
            _S["session_polygons"][_S["retake_index"]] = poly.copy()
            _S["session_scans"][_S["retake_index"]] = text
            _S["session_images"][_S["retake_index"]] = display_ready_img
            _S["retake_index"] = -1
        else:
            _S["session_polygons"].append(poly.copy())
            _S["session_scans"].append(text)
            _S["session_images"].append(display_ready_img)
            _S["scan_index"] += 1
        _S["state"] = "RESULT"
        
    elif cmd == "START_EDIT":
        idx = payload
        edited_text = show_tablet_edit_dialog(_S["session_scans"][idx])
        _S["session_scans"][idx] = edited_text
        _S["dirty"] = True
        
    elif cmd == "CMD_ADD_SCAN":
        _S["state"] = "LIVE"
        _S["roi_start"] = _S["roi_end"] = None
    elif cmd and cmd.startswith("CMD_RETAKE_"):
        _S["retake_index"] = int(cmd.split("_")[2])
        _S["state"] = "LIVE_RETAKE"
        _S["roi_start"] = _S["roi_end"] = None
    elif cmd == "CMD_UNDO":
        if _S["session_scans"]:
            _S["session_scans"].pop()
            if _S["session_images"]: _S["session_images"].pop()
            if _S["session_polygons"]: _S["session_polygons"].pop()
            _S["scan_index"] = max(1, _S["scan_index"] - 1)
        _S["state"] = "RESULT" if _S["session_scans"] else "LIVE"
    elif cmd == "CMD_VALIDATE":
        if _S["session_scans"]:
            base = _S["session_scans"][0]
            mismatches = [str(i+1) for i, t in enumerate(_S["session_scans"]) if t != base]
            if mismatches and len(_S["session_scans"]) > 1:
                _S["mismatch_info"] = f"Scan 1 does NOT match Scan(s): {', '.join(mismatches)}"
                _S["state"] = "VALIDATION_FAILED"
            else:
                _S["session_duration"] = round(time.time() - _S["session_start_time"], 1)
                _S["barcode_img"] = generate_barcode_cv2(base)
                save_ground_truth()
                _S["state"] = "BARCODE_RESULT"
    elif cmd == "CMD_NEW_SESSION":
        init_new_session()
    elif cmd == "SET_ERROR":
        _S["global_error_message"] = payload
        _S["state"] = "ERROR"

cam = WinsdkCamera() 
cv2.namedWindow('Smart Scanner', cv2.WINDOW_NORMAL)
cv2.setWindowProperty('Smart Scanner', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
cv2.setMouseCallback('Smart Scanner', mouse_handler)

live_frame_counter = 0

while True:
    loop_start = time.time()
    
    key = cv2.waitKeyEx(1)
    if key == 27: _cmd_q.put(("CMD_CANCEL", None))
    elif key == ord('q'): _cmd_q.put(("QUIT", None))

    while not _cmd_q.empty():
        c, p = _cmd_q.get_nowait()
        _handle_cmd(c, p)

    ret, live_frame = cam.read()
    if not ret or live_frame is None: continue

    fh, fw = live_frame.shape[:2]
    if fw != CAMERA_W or fh != CAMERA_H:
        CAMERA_W = fw
        CAMERA_H = fh
        display = np.zeros((CAMERA_H, CAMERA_W, 3), dtype=np.uint8)
        _S["dirty"] = True

    is_live_state = (_S["state"] in ["LIVE", "LIVE_RETAKE"])
    
    if is_live_state:
        live_frame_counter += 1
        
        # --- ENHANCEMENT: Throttled Real-Time Focus Calculation ---
        if ENABLE_FOCUS_ASSIST and live_frame_counter % FOCUS_CHECK_EVERY == 0:
            px1, py1 = max(0, (fw - PATCH_W) // 2), max(0, (fh - PATCH_H) // 2)
            px2, py2 = min(fw, px1 + PATCH_W), min(fh, py1 + PATCH_H)
            
            patch = live_frame[py1:py2, px1:px2]
            gray_patch = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
            score = cv2.Laplacian(gray_patch, cv2.CV_64F).var()
            
            _S["focus_score"] = score
            if score < FOCUS_THRESHOLD:
                _S["focus_color"] = (0, 0, 255) # Red
                _S["focus_text"] = f"HOLD STILL (Focus: {score:.1f})"
            else:
                _S["focus_color"] = (0, 255, 0) # Green
                _S["focus_text"] = f"CAPTURE NOW (Focus: {score:.1f})"
        
        if live_frame_counter % LIVE_REDRAW_EVERY == 0:
            active_buttons.clear()
            text_edit_zones.clear()
            _draw_live(display, live_frame)
    else:
        if _S["state"] in ["PROCESSING", "DETECTING"]:
            _S["dirty"] = True

        if _S["dirty"]:
            active_buttons.clear()
            text_edit_zones.clear()
            
            if _S["state"] == "SELECT_ROI": _draw_select_roi(display)
            elif _S["state"] == "DETECTING": _draw_detecting(display)
            elif _S["state"] == "SELECT_BOX": _draw_select_box(display)
            elif _S["state"] == "ADJUST_POLY": _draw_adjust_poly(display)
            elif _S["state"] == "PROCESSING": _draw_processing(display)
            elif _S["state"] == "ERROR": _draw_error(display)
            elif _S["state"] == "RESULT": _draw_result(display)
            elif _S["state"] == "VALIDATION_FAILED": _draw_validation_failed(display)
            elif _S["state"] == "BARCODE_RESULT": _draw_barcode_result(display)
            
            _S["dirty"] = False

    active_buttons.append({"rect": (CAMERA_W - 200, 20, 180, 80), "cmd": "QUIT", "label": "X CLOSE", "color": (0, 0, 200)})
    for btn in active_buttons:
        draw_button(display, btn["label"], btn["rect"], btn["color"])

    cv2.imshow('Smart Scanner', display)

    elapsed = time.time() - loop_start
    spare = FRAME_BUDGET - elapsed
    if spare > 0.002:
        time.sleep(spare)

cam.stop()
cv2.destroyAllWindows()
os._exit(0)
