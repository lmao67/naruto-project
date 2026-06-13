""" import cv2
import time
import os

wCam, hCam =640,480

cap = cv2.VideoCapture(1)
cap.set(3,wCam)
cap.set(4,hCam)

while True:
    success, img=cap.read()
    cv2.imshow("Image",img)
    cv2.waitkey(1) """



import cv2
import time
import os

wCam, hCam = 640, 480

# Changed to 0 for the default webcam
cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)

folderpath="narutoimages"
myList=os.listdir(folderpath)
print(myList)

while True:
    success, img = cap.read()
    
    # Safety check: if a frame wasn't read correctly, skip the rest of the loop
    if not success:
        print("Failed to grab frame")
        continue

    cv2.imshow("Image", img)
    
    # Added a way to break the loop by pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()