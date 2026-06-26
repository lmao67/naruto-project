import cv2
import mediapipe as mp
import csv
import os

print("Starting Two-Hand Data Collector...")

mp_hands = mp.solutions.hands
# CRITICAL: Set to 2 hands
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

# ---------------------------------------------------------
label = "The finger gun" 
# ---------------------------------------------------------

file_exists = os.path.isfile('two_hand_data.csv')

# Open a new CSV specifically for 2 hands (it needs 84 columns of math!)
with open('two_hand_data.csv', mode='a', newline='') as f:
    writer = csv.writer(f)
    
    if not file_exists:
        header = ['Label']
        # Right hand features (42 columns)
        for i in range(21): header.extend([f'R_x{i}', f'R_y{i}'])
        # Left hand features (42 columns)
        for i in range(21): header.extend([f'L_x{i}', f'L_y{i}'])
        writer.writerow(header)

    while True:
        success, frame = cap.read()
        if not success: break
        
        frame = cv2.flip(frame, 1)
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)

        # ONLY proceed if exactly TWO hands are on the screen
        if results.multi_hand_landmarks and len(results.multi_hand_landmarks) == 2:
            
            # Draw skeletons
            for hand_landmarks in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # --- THE SORTING LOGIC ---
            # Create a dictionary to safely hold our hands based on their label
            hand_dict = {'Right': None, 'Left': None}

            # Loop through the detection results and figure out which hand is which
            for i, handedness in enumerate(results.multi_handedness):
                # MediaPipe is mirrored, so we swap the labels it gives us
                hand_label = handedness.classification[0].label
                if hand_label == "Left": actual_label = "Right"
                else: actual_label = "Left"
                
                hand_dict[actual_label] = results.multi_hand_landmarks[i]

            # --- RECORDING THE DATA ---
            # Only record if we successfully found one Left and one Right hand
            if hand_dict['Right'] is not None and hand_dict['Left'] is not None:
                
                row = [label]

                # 1. Process Right Hand First
                right_hand = hand_dict['Right']
                r_wrist_x, r_wrist_y = right_hand.landmark[0].x, right_hand.landmark[0].y
                for lm in right_hand.landmark:
                    row.append(lm.x - r_wrist_x)
                    row.append(lm.y - r_wrist_y)

                # 2. Process Left Hand Second
                left_hand = hand_dict['Left']
                l_wrist_x, l_wrist_y = left_hand.landmark[0].x, left_hand.landmark[0].y
                for lm in left_hand.landmark:
                    row.append(lm.x - l_wrist_x)
                    row.append(lm.y - l_wrist_y)

                # Wait for user to press 'r' to save the frame
                key = cv2.waitKey(1) & 0xFF
                if key == ord('r'):
                    writer.writerow(row)
                    cv2.rectangle(frame, (0,0), (frame.shape[1], frame.shape[0]), (0, 255, 0), 20)
                    print(f"Recorded 1 frame of {label}!")

        # UI text
        cv2.putText(frame, f"Recording: {label}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv2.putText(frame, "Requires EXACTLY 2 hands", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
        cv2.imshow("Data Collector", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'): 
            break

cap.release()
cv2.destroyAllWindows()