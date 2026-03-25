import cv2
import pytesseract
import barcode
from barcode.writer import ImageWriter
from PIL import ImageFont
import os
import sys

# THE FONT PATCH
if not hasattr(ImageFont.FreeTypeFont, 'getsize'):
    def getsize(self, text):
        left, top, right, bottom = self.getbbox(text)
        return right - left, bottom - top
    ImageFont.FreeTypeFont.getsize = getsize

# FIND TESS PATH REGARDLESS OF THE MACHINE
def get_tesseract_path():
    if getattr(sys, 'frozen', False):
        base_dir = sys._MEIPASS
    else:
        base_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base_dir, 'Tesseract-OCR', 'tesseract.exe')

pytesseract.pytesseract.tesseract_cmd = get_tesseract_path()

# GLOBAL VARIABLES & UI STATE
drawing = False
ix, iy, ex, ey = -1, -1, -1, -1
roi_selected = False
trigger_next_state = False
trigger_reset = False
trigger_quit = False 

btn_action = [0, 0, 0, 0]
btn_reset = [0, 0, 0, 0]
btn_quit = [0, 0, 0, 0] 

state = "LIVE_1"
frozen_frame = None
val1 = ""
val2 = ""
window_name = 'Comparison Scanner'

# UI DRAWING HELPER
def draw_ui(img, action_text, show_action=True):
    global btn_action, btn_reset, btn_quit
    h, w = img.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    # UI Bar Background (optional, for better button visibility)
    # cv2.rectangle(img, (0, h - 80), (w, h), (30, 30, 30), -1)

    # Common dimensions
    bh = 60
    by = h - bh - 20
    
    # EXIT BUTTON
    qw = 120
    btn_quit = [20, by, 20 + qw, by + bh]
    cv2.rectangle(img, (btn_quit[0], btn_quit[1]), (btn_quit[2], btn_quit[3]), (40, 40, 40), -1)
    cv2.putText(img, "EXIT", (btn_quit[0]+35, btn_quit[1]+38), font, 0.6, (255, 255, 255), 2)

    # RESET BUTTON
    rw = 120
    btn_reset = [w - rw - 20, by, w - 20, by + bh]
    cv2.rectangle(img, (btn_reset[0], btn_reset[1]), (btn_reset[2], btn_reset[3]), (0, 0, 180), -1)
    cv2.putText(img, "RESET", (btn_reset[0]+25, btn_reset[1]+38), font, 0.6, (255, 255, 255), 2)

    # ACTION BUTTON
    if show_action:
        bw = 240
        btn_action = [(w // 2) - (bw // 2), by, (w // 2) + (bw // 2), by + bh]
        cv2.rectangle(img, (btn_action[0], btn_action[1]), (btn_action[2], btn_action[3]), (0, 150, 0), -1)
        t_size = cv2.getTextSize(action_text, font, 0.7, 2)[0]
        tx = btn_action[0] + (bw - t_size[0]) // 2
        ty = btn_action[1] + (bh + t_size[1]) // 2
        cv2.putText(img, action_text, (tx, ty), font, 0.7, (255, 255, 255), 2)
    else:
        btn_action = [0, 0, 0, 0]

# TOUCH/MOUSE CALLBACK
def touch_callback(event, x, y, flags, param):
    global ix, iy, ex, ey, drawing, roi_selected, trigger_next_state, trigger_reset, trigger_quit
    
    if event == cv2.EVENT_LBUTTONDOWN:
        if btn_quit[0] <= x <= btn_quit[2] and btn_quit[1] <= y <= btn_quit[3]:
            trigger_quit = True
        elif btn_reset[0] <= x <= btn_reset[2] and btn_reset[1] <= y <= btn_reset[3]:
            trigger_reset = True
        elif btn_action[0] <= x <= btn_action[2] and btn_action[1] <= y <= btn_action[3]:
            trigger_next_state = True
        else:
            drawing = True
            ix, iy = x, y
            roi_selected = False
        
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            ex, ey = x, y
            
    elif event == cv2.EVENT_LBUTTONUP:
        if drawing:
            drawing = False
            ex, ey = x, y
            if abs(ex - ix) > 15 and abs(ey - iy) > 15:
                roi_selected = True

# OCR & BARCODE LOGIC
def perform_ocr(frame, coords):
    ix, iy, ex, ey = coords
    x1, x2 = min(ix, ex), max(ix, ex)
    y1, y2 = min(iy, ey), max(iy, ey)
    roi = frame[y1:y2, x1:x2]
    if roi.size == 0: return ""
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    roi_upscaled = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    _, thresh = cv2.threshold(roi_upscaled, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    config = r'--psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-'
    text = pytesseract.image_to_string(thresh, config=config).strip()
    # return "".join(filter(str.isalnum, text)).upper()
    return "".join(text).upper().replace(" ", "")


def generate_barcode(text):
    if len(text) < 1: return None, None
    try:
        code_type = barcode.get_barcode_class('code128')
        my_barcode = code_type(text, writer=ImageWriter())
        filename = my_barcode.save("temp_barcode")
        return filename, text
    except: return None, None

# MAIN ENGINE
cap = cv2.VideoCapture(0)
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.setMouseCallback(window_name, touch_callback)

while True:
    try:
        if trigger_quit or cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
            sys.exit()
            break
    except:
        break

    ret, frame = cap.read()
    if not ret: 
        break
    h, w = frame.shape[:2]

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'): break 

    if trigger_reset or key == ord('r'):
        state = "LIVE_1"
        val1, val2 = "", ""
        ix, iy, ex, ey = -1, -1, -1, -1
        roi_selected, trigger_reset, trigger_next_state = False, False, False

    # RENDERING
    if state == "LIVE_1":
        display_frame = frame.copy()
        cv2.putText(display_frame, "Step 1: Scan Document", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        draw_ui(display_frame, "FREEZE FRAME")
        cv2.imshow(window_name, display_frame)
    
    elif state == "FROZEN_1":
        temp_view = frozen_frame.copy()
        if ix != -1: cv2.rectangle(temp_view, (ix, iy), (ex, ey), (0, 255, 0), 2)
        cv2.putText(temp_view, "Draw Box Around Batch #", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        draw_ui(temp_view, "EXTRACT DATA", show_action=roi_selected)
        cv2.imshow(window_name, temp_view)

    elif state == "LIVE_2":
        display_frame = frame.copy()
        cv2.putText(display_frame, f"Batch: {val1} | Step 2: Scan Part", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        draw_ui(display_frame, "FREEZE PART")
        cv2.imshow(window_name, display_frame)

    elif state == "FROZEN_2":
        temp_view = frozen_frame.copy()
        if ix != -1: cv2.rectangle(temp_view, (ix, iy), (ex, ey), (0, 255, 0), 2)
        cv2.putText(temp_view, "Draw Box Around Part Batch #", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        draw_ui(temp_view, "COMPARE", show_action=roi_selected)
        cv2.imshow(window_name, temp_view)

    if trigger_next_state or key == 32:
        trigger_next_state = False
        if state == "LIVE_1":
            frozen_frame = frame.copy()
            state = "FROZEN_1"
            ix, iy, ex, ey = -1, -1, -1, -1
        elif state == "FROZEN_1" and roi_selected:
            val1 = perform_ocr(frozen_frame, (ix, iy, ex, ey))
            state, roi_selected = "LIVE_2", False
            ix, iy, ex, ey = -1, -1, -1, -1
        elif state == "LIVE_2":
            frozen_frame = frame.copy()
            state = "FROZEN_2"
            ix, iy, ex, ey = -1, -1, -1, -1
        elif state == "FROZEN_2" and roi_selected:
            val2 = perform_ocr(frozen_frame, (ix, iy, ex, ey))
            result_view = frozen_frame.copy()
            cv2.rectangle(result_view, (0,0), (w, 80), (0,0,0), -1)
            
            if val1 == val2 and val1 != "":
                barcode_file, _ = generate_barcode(val1)
                if barcode_file:
                    bc_img = cv2.imread(barcode_file)
                    img_h, img_w = bc_img.shape[:2]
                    scale = min((w*0.8)/img_w, (h*0.3)/img_h)
                    new_w, new_h = int(img_w*scale), int(img_h*scale)
                    bc_resized = cv2.resize(bc_img, (new_w, new_h))
                    cv2.rectangle(result_view, (0, h-new_h-40), (w, h), (255,255,255), -1)
                    result_view[h-new_h-100:h-100, (w-new_w)//2:(w-new_w)//2+new_w] = bc_resized
                    msg, clr = f"MATCH: {val1}", (0, 255, 0)
            else:
                msg, clr = f"MISMATCH! {val1} vs {val2}", (0, 0, 255)

            cv2.putText(result_view, msg, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, clr, 2)
            state = "RESULT"
            draw_ui(result_view, "NEW SCAN")
            cv2.imshow(window_name, result_view)
        elif state == "RESULT":
            state = "LIVE_1"
            val1, val2 = "", ""

cap.release()
cv2.destroyAllWindows()
sys.exit(0)
