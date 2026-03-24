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

def perform_ocr(roi_frame):
    """Processes the pre-cut ROI and returns cleaned text"""
    if roi_frame.size == 0: return ""
    
    gray = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2GRAY)
    # Upscale for better OCR on small part numbers
    roi_upscaled = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    _, thresh = cv2.threshold(roi_upscaled, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    config = r'--psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
    text = pytesseract.image_to_string(thresh, config=config).strip()
    return "".join(filter(str.isalnum, text)).upper()

def generate_barcode(text):
    if len(text) < 1: return None
    try:
        code_type = barcode.get_barcode_class('code128')
        my_barcode = code_type(text, writer=ImageWriter())
        return my_barcode.save("temp_barcode")
    except Exception as e:
        print(f"Barcode Error: {e}")
        return None

# index of device to use for video capture
cap = cv2.VideoCapture(0)
window_name = 'Comparison Scanner'
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

# Define static capture area (Viewfinder)
# Coordinates: (x, y, width, height) - Centralized
box_w, box_h = 300,80

state = "SCAN_1"
val1 = ""
val2 = ""

while True:
    ret, frame = cap.read()
    if not ret: 
        break
    
    h, w = frame.shape[:2]

    # Calculate center for the static capture box
    cx, cy = w // 2, h // 2
    x1, y1 = cx - (box_w // 2), cy - (box_h // 2)
    x2, y2 = cx + (box_w // 2), cy + (box_h // 2)

    display_frame = frame.copy()
    
    # Draw the static viewfinder (Target Box)
    cv2.rectangle(display_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

    if state == "SCAN_1":
        msg = "1.) Align Part Number and Press SPACE"
        clr = (0, 255, 0)
    elif state == "SCAN_2":
        msg = f"Match Found: {val1} | Step 2: Align Comparison & Press SPACE"
        clr = (255, 165, 0)
    else:
        msg = "Press SPACE to reset"
        clr = (255, 255, 255)

    cv2.putText(display_frame, msg, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, clr, 2)
    cv2.imshow(window_name, display_frame)

    key = cv2.waitKey(1) & 0xFF
 
    if key == 32: # SPACEBAR
        # Capture the ROI (Region of Interest)
        roi = frame[y1:y2, x1:x2]

        if state == "SCAN_1":
            val1 = perform_ocr(roi)
            if val1:
                state = "SCAN_2"
                print(f"Captured 1: {val1}")
        
        elif state == "SCAN_2":
            val2 = perform_ocr(roi)
            print(f"Captured 2: {val2}")
            
            # Comparison Logic
            result_view = frame.copy()
            if val1 == val2 and val1 != "":
                barcode_file = generate_barcode(val1)
                res_msg = f"MATCH: {val1}"
                res_clr = (0, 255, 0)
                # Show barcode if generated
                if barcode_file:
                    bc_img = cv2.imread(barcode_file)
                    # Simple overlay logic
                    result_view[0:bc_img.shape[0], 0:bc_img.shape[1]] = bc_img
            else:
                res_msg = f"MISMATCH: {val1} vs {val2}"
                res_clr = (0, 0, 255)

            cv2.putText(result_view, res_msg, (20, h-40), cv2.FONT_HERSHEY_SIMPLEX, 1, res_clr, 3)
            cv2.imshow(window_name, result_view)
            cv2.waitKey(0) # Pause to show result
            state = "SCAN_1"

    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()