from collections import defaultdict

import cv2
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors

# Dictionary to store tracking history with default empty lists
track_history = defaultdict(lambda: [])

# Load the YOLO model with segmentation capabilities
model = YOLO("yolov8n-seg.pt")

# Open the video file
#cap = cv2.VideoCapture("ribbon.mp4")
cap = cv2.VideoCapture(0)

# Retrieve video properties: width, height, and frames per second
w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

# Initialize video writer to save the output video with the specified properties
out = cv2.VideoWriter("instance-segmentation-object-tracking.avi", cv2.VideoWriter_fourcc(*"MJPG"), fps, (w, h))

while True:
    # Read a frame from the video
    ret, im0 = cap.read()
    if not ret:
        print("Video frame is empty or video processing has been successfully completed.")
        break

    # Create an annotator object to draw on the frame
    annotator = Annotator(im0, line_width=2)

    # Perform object tracking on the current frame
    results = model.track(im0, persist=True)

    # Check if tracking IDs and masks are present in the results
    if results[0].boxes.id is not None and results[0].masks is not None:
        # Extract masks and tracking IDs
        masks = results[0].masks.xy
        track_ids = results[0].boxes.id.int().cpu().tolist()

        # Annotate each mask with its corresponding tracking ID and color
        for mask, track_id in zip(masks, track_ids):
            annotator.seg_bbox(mask=mask, mask_color=colors(track_id, True), track_label=str(track_id))

    # Write the annotated frame to the output video
    out.write(im0)
    # Display the annotated frame
    cv2.imshow("instance-segmentation-object-tracking", im0)

    # Exit the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == 27:
        break

# Release the video writer and capture objects, and close all OpenCV windows
out.release()
cap.release()
cv2.destroyAllWindows()
