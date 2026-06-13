import cv2
import mediapipe as mp
import math

print("Starting Upgraded Shadow Clone Tracker...")

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

while True:
    success, frame = cap.read()
    if not success: 
        break
        
    frame = cv2.flip(frame, 1)
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    results = hands.process(img_rgb)
    
    # We will only store the fingers IF the hand is making the correct shape
    valid_clone_hands = []

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # 1. Check the shape of THIS specific hand
            index_up = hand_landmarks.landmark[8].y < hand_landmarks.landmark[6].y
            middle_up = hand_landmarks.landmark[12].y < hand_landmarks.landmark[10].y
            ring_up = hand_landmarks.landmark[16].y < hand_landmarks.landmark[14].y
            pinky_up = hand_landmarks.landmark[20].y < hand_landmarks.landmark[18].y

            # The specific Shadow Clone shape (Index/Middle UP, Ring/Pinky DOWN)
            is_clone_shape = index_up and middle_up and not ring_up and not pinky_up

            # 2. If the hand is in the right shape, THEN save its coordinates
            if is_clone_shape:
                tip_x = int(hand_landmarks.landmark[8].x * frame.shape[1])
                tip_y = int(hand_landmarks.landmark[8].y * frame.shape[0])
                valid_clone_hands.append((tip_x, tip_y))

        # 3. Only run the collision math if EXACTLY TWO hands are making the correct shape
        if len(valid_clone_hands) == 2:
            hand_1 = valid_clone_hands[0]
            hand_2 = valid_clone_hands[1]

            distance = math.hypot(hand_2[0] - hand_1[0], hand_2[1] - hand_1[1])

            if distance < 75:  # I increased this slightly to 75 so it triggers a bit easier with 4 fingers crossed!
                cv2.putText(frame, "SHADOW CLONE JUTSU!", (50, 120), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 165, 0), 5)
            else:
                cv2.putText(frame, f"Cross them! Distance: {int(distance)}px", (50, 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    cv2.imshow("Shadow Clone Tracker", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'): 
        break

cap.release()
cv2.destroyAllWindows()