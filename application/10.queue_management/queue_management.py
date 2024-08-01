import cv2

from ultralytics import YOLO, solutions

model = YOLO("yolov8n.pt")
cap = cv2.VideoCapture("regions_video.mp4")

assert cap.isOpened(), "Error reading video file"
w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

video_writer = cv2.VideoWriter("queue_management.avi", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

queue_region = [(660, 130), (1250, 130), (1250, 800), (660, 800)]

queue = solutions.QueueManager(
    names=model.names,
    reg_pts=queue_region,
    line_thickness=3,
    fontsize=1.0,
    region_color=(255, 144, 31),
    view_img=True,
    draw_tracks=True,
    region_thickness=4
)

while cap.isOpened():
    success, im0 = cap.read()

    if success:
        tracks = model.track(im0, show=False, persist=True, verbose=False, classes=0)  # Only person class
        out = queue.process_queue(im0, tracks)

        video_writer.write(im0)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        continue

    print("Video frame is empty or video processing has been successfully completed.")
    break

cap.release()
cv2.destroyAllWindows()
