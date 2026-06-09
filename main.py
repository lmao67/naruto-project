import cv2
import mediapipe as mp
import time
import numpy as np

print("Libraries loaded! Starting camera...")

# --- VFX SETUP ---
# Load the image and keep the alpha (transparency) channel
try:
    fireball_img = cv2.imread("fireball.png", cv2.IMREAD_UNCHANGED)
    # Resize it to a good standard size (e.g., 200x200 pixels)
    fireball_img = cv2.resize(fireball_img, (200, 200))
    print("VFX Asset loaded successfully!")
except Exception as e:
    print("Warning: Could not load fireball.png. Make sure it is in the project folder!")
    fireball_img = None

def overlay_transparent(background, overlay, x, y):
    """Composites a transparent PNG over the background image at (x, y)"""
    if overlay is None: return background
    
    bg_h, bg_w, _ = background.shape
    h, w, _ = overlay.shape

    # Prevent crashing if the image goes off the edge of the screen
    if x >= bg_w or y >= bg_h or x + w <= 0 or y + h <= 0:
        return background

    # Calculate the exact pixel bounds for the crop
    x1, x2 = max(0, x), min(bg_w, x + w)
    y1, y2 = max(0, y), min(bg_h, y + h)
    overlay_x1, overlay_x2 = max(0, -x), min(w, bg_w - x)
    overlay_y1, overlay_y2 = max(0, -y), min(h, bg_h - y)

    # Separate the color and alpha channels
    overlay_rgb = overlay[overlay_y1:overlay_y2, overlay_x1:overlay_x2, :3]
    alpha_mask = overlay[overlay_y1:overlay_y2, overlay_x1:overlay_x2, 3:] / 255.0

    # Merge the pixels
    background[y1:y2, x1:x2] = (1.0 - alpha_mask) * background[y1:y2, x1:x2] + alpha_mask * overlay_rgb
    return background

# --- MEDIAPIPE SETUP ---
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

combo_step = 0  
last_sign_time = 0
combo_timeout = 3.0  
display_jutsu_time = 0  

while True:
    success, frame = cap.read()
    if not success: break
        
    frame = cv2.flip(frame, 1)
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    current_time = time.time()
    results = hands.process(img_rgb)
    
    if combo_step == 1 and (current_time - last_sign_time) > combo_timeout:
        combo_step = 0
        print("Combo timed out! Resetting to idle.")

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # We can hide the skeleton during the final Jutsu to make the VFX look cleaner
            if combo_step != 2:
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            index_up = hand_landmarks.landmark[8].y < hand_landmarks.landmark[6].y
            middle_up = hand_landmarks.landmark[12].y < hand_landmarks.landmark[10].y
            ring_up = hand_landmarks.landmark[16].y < hand_landmarks.landmark[14].y
            pinky_up = hand_landmarks.landmark[20].y < hand_landmarks.landmark[18].y

            is_tiger = index_up and middle_up and not ring_up and not pinky_up
            is_snake = not index_up and not middle_up and not ring_up and not pinky_up

            if combo_step == 0 and is_snake:
                combo_step = 1
                last_sign_time = current_time
                
            elif combo_step == 1 and is_tiger:
                if current_time - last_sign_time > 0.4: 
                    combo_step = 2
                    display_jutsu_time = current_time

            # --- RENDER THE VFX ---
            if combo_step == 2 and fireball_img is not None:
                # Get the center of the palm (landmark 9)
                palm_x = int(hand_landmarks.landmark[9].x * frame.shape[1])
                palm_y = int(hand_landmarks.landmark[9].y * frame.shape[0])
                
                # Offset the image so the center of the fireball sits on the palm
                offset_x = palm_x - (fireball_img.shape[1] // 2)
                offset_y = palm_y - (fireball_img.shape[0] // 2)

                # Paste the fireball onto the frame
                frame = overlay_transparent(frame, fireball_img, offset_x, offset_y)

    # Render UI
    if combo_step == 1:
        time_left = max(0.0, combo_timeout - (current_time - last_sign_time))
        cv2.putText(frame, f"Hit Tiger in: {time_left:.1f}s", (50, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
        
    elif combo_step == 2:
        if current_time - display_jutsu_time < 2.5:
            cv2.putText(frame, "FIREBALL JUTSU!", (50, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 4)
        else:
            combo_step = 0  

    cv2.imshow("Jutsu Tracker", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()