import cv2
from ultralytics import YOLO

model = YOLO('yolov8n-pose.pt')

cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)

while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        break

    frame = cv2.flip(frame, 1)  # 0: 상하대칭, 1 : 좌우대칭 

    results = model(frame, save=True)
    
    annotated_frame = results[0].plot() 

    scale_percent = 200 # percent of original size
    width = int(annotated_frame.shape[1] * scale_percent / 100)
    height = int(annotated_frame.shape[0] * scale_percent / 100)
    dim = (width, height)
    resized_frame = cv2.resize(annotated_frame, dim, interpolation = cv2.INTER_AREA)

    cv2.imshow('result', resized_frame)
    
    if cv2.waitKey(1) & 0xFF == 27:
      break

cap.release()
cv2.destroyAllWindows()
