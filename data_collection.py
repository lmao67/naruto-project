import cv2
import mediapipe as mp
import csv
import os

print("Starting Universal Data Collector...")

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

# ---------------------------------------------------------
label = "Tiger" # Change this for whatever you are recording
# ---------------------------------------------------------

file_exists = os.path.isfile('two_hand_data.csv')

with open('two_hand_data.csv', mode='a', newline='') as f:
    writer = csv.writer(f)
    
    if not file_exists:
        header = ['Label']
        for i in range(21): header.extend([f'R_x{i}', f'R_y{i}'])
        for i in range(21): header.extend([f'L_x{i}', f'L_y{i}'])
        writer.writerow(header)

    while True:
        success, frame = cap.read()
        if not success: break
        
        frame = cv2.flip(frame, 1)
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            hand_dict = {'Right': None, 'Left': None}

            for i, handedness in enumerate(results.multi_handedness):
                hand_label = handedness.classification[0].label
                actual_label = "Right" if hand_label == "Left" else "Left"
                hand_dict[actual_label] = results.multi_hand_landmarks[i]

            row = [label]

            # --- PROCESS RIGHT HAND ---
            if hand_dict['Right']:
                r_wrist_x, r_wrist_y = hand_dict['Right'].landmark[0].x, hand_dict['Right'].landmark[0].y
                for lm in hand_dict['Right'].landmark:
                    row.append(lm.x - r_wrist_x)
                    row.append(lm.y - r_wrist_y)
            else:
                # ZERO-PADDING: If no right hand, add 42 zeros
                row.extend([0.0] * 42)

            # --- PROCESS LEFT HAND ---
            if hand_dict['Left']:
                l_wrist_x, l_wrist_y = hand_dict['Left'].landmark[0].x, hand_dict['Left'].landmark[0].y
                for lm in hand_dict['Left'].landmark:
                    row.append(lm.x - l_wrist_x)
                    row.append(lm.y - l_wrist_y)
            else:
                # ZERO-PADDING: If no left hand, add 42 zeros
                row.extend([0.0] * 42)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('r'):
                writer.writerow(row)
                cv2.rectangle(frame, (0,0), (frame.shape[1], frame.shape[0]), (0, 255, 0), 20)
                print(f"Recorded 1 frame of {label}!")

        cv2.putText(frame, f"Recording: {label}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv2.imshow("Universal Data Collector", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'): 
            break

cap.release()
cv2.destroyAllWindows()