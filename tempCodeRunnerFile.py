if combo_step == 2 and fireball_img is not None:
                # Get the center of the palm (landmark 9)
                palm_x = int(hand_landmarks.landmark[9].x * frame.shape[1])
                palm_y = int(hand_landmarks.landmark[9].y * frame.shape[0])