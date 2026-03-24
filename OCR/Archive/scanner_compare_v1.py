import cv2
import pytesseract
import barcode
from barcode.writer import ImageWriter
from PIL import ImageFont
import numpy as np
import os
import sys

# --- 1. THE FONT PATCH ---
if not hasattr(ImageFont.FreeTypeFont, 'getsize'):
    def getsize(self, text):
        left, top, right, bottom = self.getbbox(text)
        return right - left, bottom - top
    ImageFont.FreeTypeFont.getsize = getsize

# --- 2. CONFIGURATION ---
# Ensure this path is correct for your machine
# --- 2. CONFIGURATION ---
# Ensure this path is correct for your machine
def get_tesseract_path():
    """Finds the Tesseract executable whether running as a script or .exe"""
    # Check if we are running as a compiled PyInstaller bundle
    if getattr(sys, 'frozen', False):
        # sys._MEIPASS is the temporary folder PyInstaller creates at runtime
        base_dir = sys._MEIPASS
    else:
        # If running as a normal .py script, use the folder the script is in
        base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Construct the full path to the executable
    return os.path.join(base_dir, 'Tesseract-OCR', 'tesseract.exe')

pytesseract.pytesseract.tesseract_cmd = get_tesseract_path()

# Global variables for mouse drawing
drawing = False
ix, iy = -1, -1
ex, ey = -1, -1
roi_selected = False

# TODO Transition from Freeze and Draw workflow to a static capture zone (view finder/recticle)
# better for high volume scanning 
def draw_rect(event, x, y, flags, param):
    global ix, iy, ex, ey, drawing, roi_selected
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y
        roi_selected = False
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            ex, ey = x, y
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        ex, ey = x, y
        roi_selected = True

def perform_ocr(frame, coords):
    """Helper to process ROI and return cleaned text"""
    ix, iy, ex, ey = coords
    x1, x2 = min(ix, ex), max(ix, ex)
    y1, y2 = min(iy, ey), max(iy, ey)
    
    roi = frame[y1:y2, x1:x2]
    if roi.size == 0: return ""
    
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    roi_upscaled = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    _, thresh = cv2.threshold(roi_upscaled, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # tesseract primarily trained on printed fonts
    config = r'--psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
    text = pytesseract.image_to_string(thresh, config=config).strip()
    # text = pytesseract.image_to_string(thresh).strip()
    print(text)

    return "".join(filter(str.isalnum, text)).upper()

def generate_barcode(text):
    if len(text) < 1: return None, None
    try:
        code_type = barcode.get_barcode_class('code128')
        my_barcode = code_type(text, writer=ImageWriter())
        filename = my_barcode.save("temp_barcode")
        return filename, text
    except Exception as e:
        print(f"Barcode Logic Error: {e}")
        return None, None

# index of device to use for video capture
cap = cv2.VideoCapture(0)
cv2.namedWindow('Comparison Scanner', cv2.WINDOW_NORMAL)
cv2.setMouseCallback('Comparison Scanner', draw_rect)

# STATES: LIVE_1, FROZEN_1, LIVE_2, FROZEN_2, RESULT
state = "LIVE_1"
frozen_frame = None
val1 = ""
val2 = ""

while True:
    ret, frame = cap.read()
    if not ret: break
    h, w = frame.shape[:2]

    if state == "LIVE_1":
        display_frame = frame.copy()
        cv2.putText(display_frame, "Scan Document", (20, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow('Comparison Scanner', display_frame)
    
    elif state == "FROZEN_1":
        temp_view = frozen_frame.copy()
        if ix != -1 and ex != -1:
            cv2.rectangle(temp_view, (ix, iy), (ex, ey), (0, 255, 0), 2)
        cv2.putText(temp_view, "Select Batch Number", (20, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow('Comparison Scanner', temp_view)

    elif state == "LIVE_2":
        display_frame = frame.copy()
        cv2.putText(display_frame, f"Batch Number: {val1} | Scan Part Data", (20, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.imshow('Comparison Scanner', display_frame)

    elif state == "FROZEN_2":
        temp_view = frozen_frame.copy()
        if ix != -1 and ex != -1:
            cv2.rectangle(temp_view, (ix, iy), (ex, ey), (0, 255, 0), 2)
        cv2.putText(temp_view, f"Select Batch Number From Part Data", (20, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow('Comparison Scanner', temp_view)

    key = cv2.waitKey(1) & 0xFF
 
    if key == 32: # Spacebar
        if state == "LIVE_1":
            frozen_frame = frame.copy()
            state = "FROZEN_1"
            ix, iy, ex, ey = -1, -1, -1, -1 
            
        elif state == "FROZEN_1" and roi_selected:
            val1 = perform_ocr(frozen_frame, (ix, iy, ex, ey))
            state = "LIVE_2"
            ix, iy, ex, ey = -1, -1, -1, -1
            roi_selected = False

        elif state == "LIVE_2":
            frozen_frame = frame.copy()
            state = "FROZEN_2"
            ix, iy, ex, ey = -1, -1, -1, -1

        elif state == "FROZEN_2" and roi_selected:
            val2 = perform_ocr(frozen_frame, (ix, iy, ex, ey))
            
            # 3. BUILD RESULT SCREEN
            result_view = frozen_frame.copy()
            cv2.rectangle(result_view, (0,0), (w, 60), (0,0,0), -1)
            
            if val1 == val2 and val1 != "":
                # MATCH - GENERATE BARCODE
                barcode_file, final_text = generate_barcode(val1)
                if barcode_file and os.path.exists(barcode_file):
                    bc_img = cv2.imread(barcode_file)
                    max_bc_h, max_bc_w = int(h * 0.3), int(w * 0.9)
                    img_h, img_w = bc_img.shape[:2]
                    scale = min(max_bc_w/img_w, max_bc_h/img_h)
                    new_w, new_h = int(img_w * scale), int(img_h * scale)
                    bc_resized = cv2.resize(bc_img, (new_w, new_h))
                    
                    cv2.rectangle(result_view, (0, h - max_bc_h - 40), (w, h), (255, 255, 255), -1)
                    result_view[h-new_h-20:h-20, (w-new_w)//2:(w-new_w)//2+new_w] = bc_resized
                    msg, clr = f"MATCH: {val1}", (0, 255, 0)
                else:
                    msg, clr = "OCR Failed", (0, 0, 255)
            else:
                # MISMATCH
                msg, clr = f"MISMATCH! {val1} vs {val2}", (0, 0, 255)

            cv2.putText(result_view, msg, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, clr, 2)
            cv2.imshow('Comparison Scanner', result_view)
            state = "RESULT"

        elif state == "RESULT":
            state = "LIVE_1"

    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()