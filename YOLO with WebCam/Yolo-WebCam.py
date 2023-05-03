from ultralytics import YOLO
import cv2
import cvzone
import math
import webbrowser


#for Videofile

# cap = cv2.VideoCapture("../Videos/bikes.mp4")


#WebCam Capture
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

model = YOLO("../Yolo-Weights/yolov8l.pt")

classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "Watch", "vase", "scissors",
              "teddy bear", "hair drier", "pen"
              ]

while True:
    success, img = cap.read()
    results = model(img, stream=True)

    for r in results:
        boxes = r.boxes
        for b in boxes:

            #bounding boxes
            x1,y1,x2,y2 = b.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            cv2.rectangle(img, (x1, y1), (x2, y2), (245, 0, 255), 3)

            conf = math.ceil((b.conf[0]*100))/100 #confidance calculate
            #print(conf) #display confidance

            #class name
            if conf > 0.5:
                cls = int(b.cls[0])
                cvzone.putTextRect(img, f'{classNames[cls]} {conf}', (max(0, x1), max(40, y1)),scale=1,thickness=1)

            #ob = classNames[cls]



    cv2.imshow("Object Detetction",img)
    cv2.waitKey(1)
