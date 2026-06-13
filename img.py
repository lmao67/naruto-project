import cv2
""" img=cv2.imread("narutoimages/fireball.png")
cv2.imshow("Basit",img)
cv2.waitKey(0) """
frameWidth=640
frameHeight=480
cap=cv2.VideoCapture("narutoimages/rnddd.mp4")
#cap.set(3,frameWidth)
#cap.set(4,frameHeight)

while True:
    sucess,img=cap.read()
    img=cv2.resize(img,(frameWidth,frameHeight))
    cv2.imshow("Video",img)

    if cv2.waitKey(1) & 0XFF == ord('q'):
        break