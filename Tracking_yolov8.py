import cv2
from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO('yolov8n-pose.pt')

# Open the video file
#video_path = "path/to/video.mp4"
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        frame = cv2.flip(frame, 1)
        
        # Run YOLOv8 tracking on the frame, persisting tracks between frames
        results = model.track(frame, persist=True)

        # Visualize the results on the frame
        annotated_frame = results[0].plot()

        scale_percent = 200 # percent of original size
        width = int(annotated_frame.shape[1] * scale_percent / 100)
        height = int(annotated_frame.shape[0] * scale_percent / 100)
        dim = (width, height)
        resized_frame = cv2.resize(annotated_frame, dim, interpolation = cv2.INTER_AREA)

        # Display the annotated frame
        cv2.imshow("YOLOv8 Tracking", resized_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == 27:
            break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()
