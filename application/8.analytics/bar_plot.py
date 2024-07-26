import cv2

from ultralytics import YOLO, solutions

model = YOLO("yolov8s.pt")

cap = cv2.VideoCapture("street.mp4")
assert cap.isOpened(), "Error reading video file"
w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

out = cv2.VideoWriter("bar_plot.avi", cv2.VideoWriter_fourcc(*"MJPG"), fps, (w, h))

analytics = solutions.Analytics(
    type="bar",
    writer=out,
    im0_shape=(w, h),
    view_img=True,
)

clswise_count = {}

while cap.isOpened():
    success, frame = cap.read()
    if success:
        results = model.track(frame, persist=True, verbose=True)
        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu()
            clss = results[0].boxes.cls.cpu().tolist()
            for box, cls in zip(boxes, clss):
                if model.names[int(cls)] in clswise_count:
                    clswise_count[model.names[int(cls)]] += 1
                else:
                    clswise_count[model.names[int(cls)]] = 1

            analytics.update_bar(clswise_count)
            clswise_count = {}

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        break

cap.release()
out.release()
cv2.destroyAllWindows()
