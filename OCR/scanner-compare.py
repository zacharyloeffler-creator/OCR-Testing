import os
import sys
import cv2
import pytesseract
import barcode
from barcode.writer import ImageWriter
from PIL import ImageFont


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


def on_zoom_change(val):
    global box_size
    # Ensure the box doesn't get too small or larger than the frame
    box_size = max(50, val)


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
cv2.createTrackbar('ROI Zoom', window_name, 200, 600, on_zoom_change)

box_size = 400
state = "SCAN_1"
val1 = ""
val2 = ""

while True:
    ret, frame = cap.read()
    
    # break conditions
    if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1 or not ret:
        break
    
    h, w = frame.shape[:2]

    curr_w = box_size
    curr_h = int(box_size / 3)

    # Calculate center for the static capture box
    cx, cy = w // 2, h // 2
    x1, y1 = cx - (curr_w // 2), cy - (curr_h // 2)
    x2, y2 = cx + (curr_w // 2), cy + (curr_h // 2)

    display_frame = frame.copy()
    
    # Draw the static viewfinder (Target Box)
    cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
    
    # UI Overlay
    cv2.putText(display_frame, f"State: {state}", (20, 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(display_frame, "Adjust slider to fit text", (20, 70), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

    cv2.imshow(window_name, display_frame)

    key = cv2.waitKey(1) & 0xFF
 
    if key == 32: # SPACEBAR
        # get region of interest to perform ocr on 
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
            cv2.waitKey(0)
            state = "SCAN_1"
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()