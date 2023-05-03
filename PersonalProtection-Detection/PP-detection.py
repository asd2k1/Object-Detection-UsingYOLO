from ultralytics import YOLO
import cv2
import cvzone
import math
#for Videofile

#cap = cv2.VideoCapture("../Videos/ppe-1.mp4")
#WebCam Capture


cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

model = YOLO("ppdetection.pt")

classNames = ['Hardhat', 'Mask', 'NO-Hardhat', 'NO-Mask', 'NO-Safety Vest', 'Person', 'Safety Cone', 'Safety Vest', 'machinery', 'vehicle']

rColor = (0,0,255)

while True:
    success, img = cap.read()
    results = model(img, stream=True)

    for r in results:
        boxes = r.boxes
        for b in boxes:
            #bounding boxes
            x1,y1,x2,y2 = b.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)


            conf = math.ceil((b.conf[0]*100))/100 #confidance calculate
            #print(conf)

            #class name
            cls = int(b.cls[0])
            cvzone.putTextRect(img, f'{classNames[cls]} {conf}', (max(0, x1), max(40, y1)),scale=1,thickness=1)

            #Change color on detetction

            currentClass = classNames[cls]
            if conf>0.5:
                if currentClass == 'NO-Hardhat' or currentClass == 'NO-Safety Vest' or currentClass == 'NO-Mask':
                    rColor = (0,0,255)
                elif currentClass == 'Hardhat' or currentClass == 'Safety Vest' or currentClass == 'Mask':
                    rColor = (0,255,0)

                else:
                    rColor = (255,0,0)
            cv2.rectangle(img, (x1, y1), (x2, y2), rColor, 3)

    cv2.imshow("Personal Protection Kit Detection",img)
    cv2.waitKey(1)
