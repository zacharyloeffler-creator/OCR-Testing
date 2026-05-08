import cv2
import easyocr
import numpy as np
import threading
import time
import io

# --- NEW IMPORTS FOR BARCODE ---
try:
    import barcode
    from barcode.writer import ImageWriter
except ImportError:
    print("WARNING: python-barcode not found. Please run: pip install python-barcode")

# --- 1. CONFIGURATION & GLOBALS ---
print("Initializing EasyOCR... (Using CPU)")
reader = easyocr.Reader(['en'], gpu=False)

# States: LIVE, SELECT_ROI, DETECTING, SELECT_BOX, ADJUST_POLY, PROCESSING, RESULT, VALIDATION_FAILED, BARCODE_RESULT
state = "LIVE"  
frame = None
frozen_frame = None
detected_boxes = []

# --- NEW SESSION GLOBALS ---
session_scans = []
mismatch_info = ""
barcode_img = None

# UI Command Queue 
pending_command = None
active_buttons = []

# ROI Selection
roi_start = None
roi_end = None
roi_drawing = False

# Polygon adjustment
selected_polygon = None
dragging_point_idx = -1
hover_point_idx = -1
DRAG_RADIUS = 40 
zoom_params = {}

extracted_text = ""
debug_warped = None

# --- 2. THREADED CAMERA CLASS ---
class ThreadedCamera:
    def __init__(self, src=0, width=1920, height=1080):
        self.cap = cv2.VideoCapture(src, cv2.CAP_MSMF)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.ret, self.frame = self.cap.read()
        self.stopped = False
        threading.Thread(target=self.update, daemon=True).start()

    def update(self):
        while not self.stopped:
            ret, frame = self.cap.read()
            if ret:
                self.ret = ret
                self.frame = frame
            time.sleep(0.01)

    def read(self):
        return self.ret, self.frame.copy() if self.ret else None

    def stop(self):
        self.stopped = True
        self.cap.release()

# --- 3. BACKGROUND WORKER THREADS ---
def run_detection_thread(img, rx1, ry1, rx2, ry2):
    global detected_boxes, state
    crop = img[ry1:ry2, rx1:rx2]
    horizontal_list, _ = reader.detect(crop)
    
    detected_boxes = []
    if horizontal_list:
        for box in horizontal_list[0]:
            xmin, xmax, ymin, ymax = box
            detected_boxes.append([xmin + rx1, xmax + rx1, ymin + ry1, ymax + ry1])
    state = "SELECT_BOX"


def run_ocr_thread(img, polygon_pts):
    global extracted_text, debug_warped, state, session_scans
    warped = four_point_transform(img, polygon_pts)
    if warped.size == 0 or warped.shape[0] < 5 or warped.shape[1] < 5:
        extracted_text = "ERROR: Invalid Polygon"
        state = "RESULT"
        return

    # Keep the grayscale conversion and upscale
    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    upscaled = cv2.resize(gray, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    
    # REMOVE the cv2.adaptiveThreshold line. 
    debug_warped = upscaled.copy() 
    
    invoice_chars = '0123456789ABCDEFGHIJKLMNPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz-#/.'
    
    # Pass the 'upscaled' image directly, and let EasyOCR handle the contrast
    results = reader.readtext(
        upscaled,  
        detail=0, 
        paragraph=False, 
        decoder='beamsearch',
        beamWidth=10,            
        mag_ratio=1.0,           
        allowlist=invoice_chars, 
        # Uncomment these to let EasyOCR's AI handle poor lighting naturally:
        contrast_ths=0.05,      
        adjust_contrast=0.5       
    )
    extracted_text = " ".join(results).strip()
    
    if extracted_text and not extracted_text.startswith("ERROR"):
        session_scans.append(extracted_text)

    state = "RESULT"

## OLD ###
# def run_ocr_thread(img, polygon_pts):
#     global extracted_text, debug_warped, state, session_scans
#     warped = four_point_transform(img, polygon_pts)
#     if warped.size == 0 or warped.shape[0] < 5 or warped.shape[1] < 5:
#         extracted_text = "ERROR: Invalid Polygon"
#         state = "RESULT"
#         return

#     gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
#     upscaled = cv2.resize(gray, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
#     binary = cv2.adaptiveThreshold(upscaled, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 5)
#     debug_warped = binary.copy()
    
#     invoice_chars = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz-#/.'
#     # results = reader.readtext(binary, detail=0, paragraph=False, decoder='greedy', mag_ratio=1.0)
#     results = reader.readtext(
#         binary, 
#         detail=0, 
#         paragraph=False, 
#         decoder='beamsearch',
#         #rotation_info=[90, 180, 270],
#         beamWidth=10,             # More thorough search for character sequences
#         mag_ratio=1.0,            # Slight zoom for better small-text detection
#         allowlist=invoice_chars,  # Restrict to relevant characters only
#         #contrast_ths=0.05,        # Be aggressive with low-contrast scans
#         #add_margin=0.2            # Give the text some "breathing room"
#     )
#     extracted_text = " ".join(results).strip()
    
#     # Append to current session
#     if extracted_text and not extracted_text.startswith("ERROR"):
#         session_scans.append(extracted_text)
        
#     state = "RESULT"

# --- 4. MATH, WARPING & BARCODE HELPERS ---
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
    """Generates a Code128 barcode and converts it to an OpenCV compatible numpy array."""
    try:
        rv = io.BytesIO()
        code128 = barcode.get('code128', text, writer=ImageWriter())
        code128.write(rv, options={"write_text": True})
        rv.seek(0)
        img_arr = np.frombuffer(rv.read(), dtype=np.uint8)
        img = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
        return img
    except Exception as e:
        print(f"Failed to generate barcode: {e}")
        # Return a red error block if generation fails
        err_img = np.zeros((200, 600, 3), dtype=np.uint8)
        err_img[:] = (0, 0, 255)
        cv2.putText(err_img, "BARCODE GEN ERROR", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        return err_img

# --- 5. UI & MOUSE HANDLER ---
def draw_button(img, text, rect, bg_color, text_color=(255, 255, 255)):
    x, y, w, h = rect
    cv2.rectangle(img, (x, y), (x + w, y + h), bg_color, -1)
    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 255), 2) 
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.0
    thickness = 2
    text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
    txt_x = x + (w - text_size[0]) // 2
    txt_y = y + (h + text_size[1]) // 2
    cv2.putText(img, text, (txt_x, txt_y), font, font_scale, text_color, thickness)

def mouse_handler(event, x, y, flags, param):
    global state, selected_polygon, dragging_point_idx, hover_point_idx, zoom_params
    global roi_start, roi_end, roi_drawing, pending_command, active_buttons

    if event == cv2.EVENT_LBUTTONDOWN:
        for btn in active_buttons:
            bx, by, bw, bh = btn["rect"]
            if bx <= x <= bx + bw and by <= y <= by + bh:
                pending_command = btn["cmd"]
                return 

    if state == "SELECT_ROI":
        if event == cv2.EVENT_LBUTTONDOWN:
            roi_start = (x, y)
            roi_end = (x, y)
            roi_drawing = True
        elif event == cv2.EVENT_MOUSEMOVE and roi_drawing:
            roi_end = (x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            roi_end = (x, y)
            roi_drawing = False

    elif state == "SELECT_BOX":
        if event == cv2.EVENT_LBUTTONDOWN:
            for box in detected_boxes:
                xmin, xmax, ymin, ymax = box
                if xmin <= x <= xmax and ymin <= y <= ymax:
                    selected_polygon = np.array([
                        [xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]
                    ], dtype="float32")
                    
                    padding = 100
                    h, w = frozen_frame.shape[:2]
                    zx1, zy1 = max(0, int(xmin) - padding), max(0, int(ymin) - padding)
                    zx2, zy2 = min(w, int(xmax) + padding), min(h, int(ymax) + padding)
                    
                    scale = min(w / (zx2 - zx1), h / (zy2 - zy1))
                    scaled_w, scaled_h = int((zx2 - zx1) * scale), int((zy2 - zy1) * scale)
                    
                    zoom_params = {
                        "zx1": zx1, "zy1": zy1, "zx2": zx2, "zy2": zy2,
                        "scale": scale, "off_x": (w - scaled_w) // 2, "off_y": (h - scaled_h) // 2
                    }
                    state = "ADJUST_POLY"
                    break

    elif state == "ADJUST_POLY":
        hover_point_idx = -1
        zp = zoom_params
        ix = (x - zp["off_x"]) / zp["scale"] + zp["zx1"]
        iy = (y - zp["off_y"]) / zp["scale"] + zp["zy1"]
        
        if selected_polygon is not None:
            for i, pt in enumerate(selected_polygon):
                sx = (pt[0] - zp["zx1"]) * zp["scale"] + zp["off_x"]
                sy = (pt[1] - zp["zy1"]) * zp["scale"] + zp["off_y"]
                if np.sqrt((sx - x) ** 2 + (sy - y) ** 2) < DRAG_RADIUS:
                    hover_point_idx = i
                    break

        if event == cv2.EVENT_LBUTTONDOWN and hover_point_idx != -1:
            dragging_point_idx = hover_point_idx
        elif event == cv2.EVENT_MOUSEMOVE and dragging_point_idx != -1:
            selected_polygon[dragging_point_idx] = [ix, iy]
        elif event == cv2.EVENT_LBUTTONUP:
            dragging_point_idx = -1

# --- 6. MAIN LOOP ---
cam = ThreadedCamera(src=0, width=1920, height=1080) 

cv2.namedWindow('Smart Scanner', cv2.WINDOW_NORMAL)
cv2.setWindowProperty('Smart Scanner', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
cv2.setMouseCallback('Smart Scanner', mouse_handler)

while True:
    # --- PROCESS UI COMMANDS ---
    key = cv2.waitKey(1) & 0xFF
    cmd = pending_command
    pending_command = None 
    
    if key == 27: cmd = "CMD_CANCEL"
    elif key == ord('q'): cmd = "QUIT"

    if cmd == "QUIT":
        break
    elif cmd == "CMD_CANCEL":
        if state in ["SELECT_ROI", "SELECT_BOX", "ADJUST_POLY"]:
            state = "LIVE"
            roi_start = roi_end = None
            
    elif cmd == "CMD_ACTION_1": # General Action 1 (Usually 'Capture')
        if state == "LIVE":
            frozen_frame = cam.frame.copy() if cam.frame is not None else None
            if frozen_frame is not None:
                state = "SELECT_ROI"
                roi_start = roi_end = None
                
    elif cmd == "CMD_ACTION_2": # General Action 2 (Usually 'Confirm')
        if state == "SELECT_ROI":
            if roi_start and roi_end and abs(roi_start[0] - roi_end[0]) > 10 and abs(roi_start[1] - roi_end[1]) > 10:
                rx1, ry1 = min(roi_start[0], roi_end[0]), min(roi_start[1], roi_end[1])
                rx2, ry2 = max(roi_start[0], roi_end[0]), max(roi_start[1], roi_end[1])
                
                state = "DETECTING"
                threading.Thread(target=run_detection_thread, args=(frozen_frame, rx1, ry1, rx2, ry2), daemon=True).start()
            else:
                print("Action blocked: Please drag to select a valid ROI first.")
                
        elif state == "ADJUST_POLY":
            state = "PROCESSING"
            threading.Thread(target=run_ocr_thread, args=(frozen_frame, selected_polygon), daemon=True).start()

    # --- NEW SESSION COMMANDS ---
    elif cmd == "CMD_ADD_SCAN":
        state = "LIVE"
        roi_start = roi_end = None
        
    elif cmd == "CMD_UNDO":
        if session_scans:
            session_scans.pop() # Remove the last scan
        if not session_scans: # If session is now empty, go back to LIVE
            state = "LIVE"
        else:
            state = "RESULT"
            
    elif cmd == "CMD_VALIDATE":
        if len(session_scans) > 1:
            base_text = session_scans[0]
            # Find mismatched indices (compare against Scan 1)
            mismatches = [str(i+1) for i, text in enumerate(session_scans) if text != base_text]
            
            for i,text in enumerate(session_scans):
                print(text)

            if mismatches:
                mismatch_info = f"Scan 1 does NOT match Scan(s): {', '.join(mismatches)}"
                state = "VALIDATION_FAILED"
            else:
                # All match! Generate barcode
                barcode_img = generate_barcode_cv2(base_text)
                state = "BARCODE_RESULT"
                
    elif cmd == "CMD_NEW_SESSION":
        session_scans.clear()
        barcode_img = None
        state = "LIVE"
        roi_start = roi_end = None

    # --- DRAWING PHASE ---
    ret, frame = cam.read()
    if not ret or frame is None:
        continue
        
    display = frame.copy() if state == "LIVE" else (frozen_frame.copy() if frozen_frame is not None else frame.copy())
    h, w = display.shape[:2]
    active_buttons = [] 

    btn_y = h - 120
    btn_h = 80

    if state == "LIVE":
        scan_count = len(session_scans)
        prompt = "Aim Camera at Document" if scan_count == 0 else f"Scan #{scan_count + 1} - Aim Camera"
        cv2.putText(display, prompt, (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
        active_buttons.append({"rect": (w//2 - 150, btn_y, 300, btn_h), "cmd": "CMD_ACTION_1", "label": "CAPTURE", "color": (0, 200, 0)})

    elif state == "SELECT_ROI":
        overlay = display.copy()
        cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
        display = cv2.addWeighted(overlay, 0.4, display, 0.6, 0)

        if roi_start and roi_end:
            x1, y1 = min(roi_start[0], roi_end[0]), min(roi_start[1], roi_end[1])
            x2, y2 = max(roi_start[0], roi_end[0]), max(roi_start[1], roi_end[1])
            if x2 > x1 and y2 > y1:
                display[y1:y2, x1:x2] = frozen_frame[y1:y2, x1:x2]
                cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 255), 3)
                
        cv2.putText(display, "Drag across text, then tap CONFIRM", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
        active_buttons.append({"rect": (50, btn_y, 200, btn_h), "cmd": "CMD_CANCEL", "label": "CANCEL", "color": (0, 0, 200)})
        active_buttons.append({"rect": (w - 350, btn_y, 300, btn_h), "cmd": "CMD_ACTION_2", "label": "CONFIRM ROI", "color": (0, 200, 0)})

    elif state == "DETECTING":
        dots = "." * (int(time.time() * 3) % 4)
        cv2.putText(display, f"DETECTING TEXT{dots}", (w//2 - 200, h//2), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 4)

    elif state == "SELECT_BOX":
        if roi_start and roi_end:
            rx1, ry1 = min(roi_start[0], roi_end[0]), min(roi_start[1], roi_end[1])
            rx2, ry2 = max(roi_start[0], roi_end[0]), max(roi_start[1], roi_end[1])
            cv2.rectangle(display, (rx1, ry1), (rx2, ry2), (0, 165, 255), 2)
            
        for box in detected_boxes:
            xmin, xmax, ymin, ymax = [int(v) for v in box]
            cv2.rectangle(display, (xmin, ymin), (xmax, ymax), (0, 255, 0), 3)
            
        cv2.putText(display, "Tap a green box to select it", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
        active_buttons.append({"rect": (50, btn_y, 200, btn_h), "cmd": "CMD_CANCEL", "label": "BACK", "color": (0, 0, 200)})

    elif state == "ADJUST_POLY":
        display = np.zeros_like(frozen_frame)
        zp = zoom_params
        
        crop = frozen_frame[zp["zy1"]:zp["zy2"], zp["zx1"]:zp["zx2"]]
        scaled_crop = cv2.resize(crop, None, fx=zp["scale"], fy=zp["scale"])
        display[zp["off_y"]:zp["off_y"]+scaled_crop.shape[0], zp["off_x"]:zp["off_x"]+scaled_crop.shape[1]] = scaled_crop
        
        pts_screen = []
        for pt in selected_polygon:
            sx = int((pt[0] - zp["zx1"]) * zp["scale"] + zp["off_x"])
            sy = int((pt[1] - zp["zy1"]) * zp["scale"] + zp["off_y"])
            pts_screen.append([sx, sy])
            
        pts_screen = np.array(pts_screen)
        cv2.polylines(display, [pts_screen], isClosed=True, color=(255, 0, 0), thickness=3)
        
        for i, pt in enumerate(pts_screen):
            color = (0, 0, 255) if i == hover_point_idx else (0, 255, 0)
            cv2.circle(display, tuple(pt), 12, color, -1) 
            cv2.circle(display, tuple(pt), 14, (255,255,255), 2) 
            
        if dragging_point_idx != -1:
            ix, iy = int(selected_polygon[dragging_point_idx][0]), int(selected_polygon[dragging_point_idx][1])
            patch_size = 80
            half = patch_size // 2
            padded = cv2.copyMakeBorder(frozen_frame, half, half, half, half, cv2.BORDER_REPLICATE)
            px, py = ix + half, iy + half
            patch = padded[py - half : py + half, px - half : px + half]
            
            mag_scale = 3
            magnified = cv2.resize(patch, None, fx=mag_scale, fy=mag_scale, interpolation=cv2.INTER_NEAREST)
            mh, mw = magnified.shape[:2]
            
            cv2.line(magnified, (mw//2, 0), (mw//2, mh), (0, 255, 0), 2)
            cv2.line(magnified, (0, mh//2), (mw, mh//2), (0, 255, 0), 2)
            
            tr_x, tr_y = w - mw - 40, 40
            display[tr_y : tr_y + mh, tr_x : tr_x + mw] = magnified
            cv2.rectangle(display, (tr_x, tr_y), (tr_x + mw, tr_y + mh), (255, 255, 255), 4)

        cv2.putText(display, "Drag corners to adjust, then tap CONFIRM", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 3)
        active_buttons.append({"rect": (50, btn_y, 200, btn_h), "cmd": "CMD_CANCEL", "label": "BACK", "color": (0, 0, 200)})
        active_buttons.append({"rect": (w - 300, btn_y, 250, btn_h), "cmd": "CMD_ACTION_2", "label": "CONFIRM", "color": (0, 200, 0)})

    elif state == "PROCESSING":
        dots = "." * (int(time.time() * 4) % 4)
        cv2.putText(display, f"READING TEXT{dots}", (w//2 - 200, h//2), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 4)

    # --- NEW: SESSION RESULT STATE ---
    elif state == "RESULT":
        cv2.rectangle(display, (0, 0), (w, h), (0, 0, 0), -1) # Darken entire screen for easier list reading
        cv2.putText(display, f"SESSION SCANS (Total: {len(session_scans)})", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
        
        # Display all scans recorded so far
        for i, text in enumerate(session_scans):
            y_pos = 140 + (i * 60)
            cv2.putText(display, f"Scan {i+1}: {text}", (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)

        # Generate Button Layout based on array size
        btn_w = 250
        gap = 50
        num_buttons = 3 if len(session_scans) > 1 else 2
        start_x = (w - (num_buttons * btn_w + (num_buttons - 1) * gap)) // 2
        
        active_buttons.append({"rect": (start_x, btn_y, btn_w, btn_h), "cmd": "CMD_ADD_SCAN", "label": "+ ADD SCAN", "color": (200, 100, 0)})
        active_buttons.append({"rect": (start_x + btn_w + gap, btn_y, btn_w, btn_h), "cmd": "CMD_UNDO", "label": "UNDO LAST", "color": (0, 0, 200)})
        
        if len(session_scans) > 1:
             active_buttons.append({"rect": (start_x + 2*(btn_w + gap), btn_y, btn_w, btn_h), "cmd": "CMD_VALIDATE", "label": "VALIDATE", "color": (0, 200, 0)})

    # --- NEW: VALIDATION FAILED STATE ---
    elif state == "VALIDATION_FAILED":
        cv2.rectangle(display, (0, 0), (w, h), (0, 0, 0), -1)
        cv2.putText(display, "VALIDATION FAILED!", (w//2 - 300, h//2 - 100), cv2.FONT_HERSHEY_SIMPLEX, 1.8, (0, 0, 255), 5)
        cv2.putText(display, mismatch_info, (w//2 - 450, h//2), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
        
        active_buttons.append({"rect": (w//2 - 300, btn_y, 250, btn_h), "cmd": "CMD_UNDO", "label": "UNDO MISTAKE", "color": (0, 0, 200)})
        active_buttons.append({"rect": (w//2 + 50, btn_y, 250, btn_h), "cmd": "CMD_NEW_SESSION", "label": "DISCARD ALL", "color": (100, 100, 100)})

    # --- NEW: BARCODE RESULT STATE ---
    elif state == "BARCODE_RESULT":
        cv2.rectangle(display, (0, 0), (w, h), (0, 0, 0), -1)
        cv2.putText(display, "VALIDATION SUCCESSFUL!", (w//2 - 350, 120), cv2.FONT_HERSHEY_SIMPLEX, 1.8, (0, 255, 0), 5)
        
        if barcode_img is not None:
            bh, bw = barcode_img.shape[:2]
            bx = (w - bw) // 2
            by = (h - bh) // 2
            # Handle potential overflow if barcode is unusually huge
            if by > 0 and bx > 0 and by+bh < h and bx+bw < w:
                display[by:by+bh, bx:bx+bw] = barcode_img
            else:
                cv2.putText(display, "Barcode too large to preview", (w//2 - 200, h//2), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
            
        active_buttons.append({"rect": (w//2 - 150, btn_y, 300, btn_h), "cmd": "CMD_NEW_SESSION", "label": "NEW SESSION", "color": (0, 200, 0)})

    # --- RENDER UI BUTTONS ON TOP ---
    # NEW: Add a universal EXIT button in the top-right corner
    exit_btn_w, exit_btn_h = 120, 60
    active_buttons.append({
        "rect": (w - exit_btn_w - 20, 20, exit_btn_w, exit_btn_h), 
        "cmd": "QUIT", 
        "label": "EXIT", 
        "color": (0, 0, 200) # Red background
    })

    for btn in active_buttons:
        draw_button(display, btn["label"], btn["rect"], btn["color"])

    cv2.imshow('Smart Scanner', display)

cam.stop()
cv2.destroyAllWindows()
