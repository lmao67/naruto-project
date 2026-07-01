import cv2
import mediapipe as mp
import pandas as pd
import pickle
import time
from collections import deque

print("Powering up the Jutsu Engine...")

# --- 1. LOAD YOUR AI BRAIN ---
try:
    with open('jutsu_model.pkl', 'rb') as f:
        model = pickle.load(f)
    print("AI Brain loaded successfully!")
except FileNotFoundError:
    print("Error: Could not find jutsu_model.pkl!")
    exit()

# --- 2. LOAD VISUAL EFFECTS ---
def load_vfx(filename):
    img = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
    if img is not None:
        return cv2.resize(img, (300, 300))
    else:
        print(f"Warning: Could not load image at {filename}. Check the folder name and spelling!")
        return None

# UPDATED: Pointing to your new 'narutoimages' folder
vfx_assets = {
    "Fireball": load_vfx("narutoimages/fireball.png")
}

def overlay_transparent(bg, overlay, x, y):
    if overlay is None: return bg
    bg_h, bg_w, _ = bg.shape
    h, w, _ = overlay.shape
    if x >= bg_w or y >= bg_h or x + w <= 0 or y + h <= 0: return bg
    x1, x2 = max(0, x), min(bg_w, x + w)
    y1, y2 = max(0, y), min(bg_h, y + h)
    overlay_x1, overlay_x2 = max(0, -x), min(w, bg_w - x)
    overlay_y1, overlay_y2 = max(0, -y), min(h, bg_h - y)
    overlay_rgb = overlay[overlay_y1:overlay_y2, overlay_x1:overlay_x2, :3]
    alpha_mask = overlay[overlay_y1:overlay_y2, overlay_x1:overlay_x2, 3:] / 255.0
    bg[y1:y2, x1:x2] = (1.0 - alpha_mask) * bg[y1:y2, x1:x2] + alpha_mask * overlay_rgb
    return bg

# --- 3. THE COMBO SYSTEM ---
COMBO_RECIPES = {
    ("Snake", "Tiger"): "Fireball",
}

combo_buffer = deque(maxlen=3) # Remembers your last 3 signs
active_vfx = None
vfx_timer = 0
last_sign = None

# --- 4. CAMERA & TRACKING SETUP ---
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils
cap = cv2.VideoCapture(0)

print("\n--- SYSTEM ONLINE. AWAITING HAND SIGNS... ---")

while True:
    success, frame = cap.read()
    if not success: break
    frame = cv2.flip(frame, 1)
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)
    
    current_time = time.time()

    if results.multi_hand_landmarks:
        hand_dict = {'Right': None, 'Left': None}
        for i, handedness in enumerate(results.multi_handedness):
            # MediaPipe is mirrored, so we swap Left/Right labels
            actual_label = "Right" if handedness.classification[0].label == "Left" else "Left"
            hand_dict[actual_label] = results.multi_hand_landmarks[i]
            mp_draw.draw_landmarks(frame, results.multi_hand_landmarks[i], mp_hands.HAND_CONNECTIONS)

        # --- EXTRACT 84 COLUMNS OF DATA FOR THE AI ---
        row = []
        
        # Right Hand (42 points or 42 zeros)
        if hand_dict['Right']:
            rx, ry = hand_dict['Right'].landmark[0].x, hand_dict['Right'].landmark[0].y
            for lm in hand_dict['Right'].landmark:
                row.extend([lm.x - rx, lm.y - ry])
        else:
            row.extend([0.0] * 42)

        # Left Hand (42 points or 42 zeros)
        if hand_dict['Left']:
            lx, ly = hand_dict['Left'].landmark[0].x, hand_dict['Left'].landmark[0].y
            for lm in hand_dict['Left'].landmark:
                row.extend([lm.x - lx, lm.y - ly])
        else:
            row.extend([0.0] * 42)

        # --- LET THE AI PREDICT THE SIGN ---
        df_row = pd.DataFrame([row], columns=[f'f_{i}' for i in range(84)]) 
        prediction = model.predict(df_row)[0]
        
        cv2.putText(frame, f"AI Sees: {prediction}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # --- COMBO LOGIC ---
        if prediction != last_sign and prediction != "Idle":  
            combo_buffer.append(prediction)
            last_sign = prediction
            
            for sequence, jutsu_name in COMBO_RECIPES.items():
                if tuple(combo_buffer)[-len(sequence):] == sequence:
                    print(f"JUTSU TRIGGERED: {jutsu_name}!!!")
                    active_vfx = jutsu_name
                    vfx_timer = current_time
                    combo_buffer.clear() 
                    break
        elif prediction == "Idle":
            last_sign = "Idle"

        # --- RENDER VISUAL EFFECTS ---
        if active_vfx and (current_time - vfx_timer < 2.5): 
            cv2.putText(frame, f"{active_vfx.upper()}!!!", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 4)
            
            vfx_img = vfx_assets.get("Fireball")
            
            # Draw the fireball on your right hand
            if vfx_img is not None and hand_dict['Right']:
                palm_x = int(hand_dict['Right'].landmark[9].x * frame.shape[1])
                palm_y = int(hand_dict['Right'].landmark[9].y * frame.shape[0])
                frame = overlay_transparent(frame, vfx_img, palm_x - 150, palm_y - 150)
        else:
            active_vfx = None

    cv2.putText(frame, f"Combo Memory: {list(combo_buffer)}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    cv2.imshow("Real-Time Jutsu Engine", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'): 
        break

cap.release()
cv2.destroyAllWindows()