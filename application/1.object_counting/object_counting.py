import cv2

from ultralytics import YOLO, solutions

# Load the pre-trained YOLOv8 model
model = YOLO("yolov8n.pt")

# Open the video file
cap = cv2.VideoCapture("test2.mp4")
assert cap.isOpened(), "Error reading video file"

# Get video properties: width, height, and frames per second (fps)
w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

# Define points for a line or region of interest in the video frame
#region_points = [(216, 415), (1130, 415), (1130, 445), (216, 445)]   # test1.mp4 region points
region_points = [(200, 500), (1050, 500), (1050, 460), (200, 460)]   # test2.mp4 region points

# Define region points as a polygon with 5 points
#region_points = [(20, 400), (1080, 404), (1080, 360), (20, 360), (20, 400)]

# Define line points
#region_points = [(0, 750), (1920, 750)]   # test3.mp4 line points

# Specify classes to count, for example: person (0) and car (2)
classes_to_count = [0, 2]  # Class IDs for person and car

# Initialize the video writer to save the output video
video_writer = cv2.VideoWriter("object_counting_output2.avi", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

# Initialize the Object Counter with visualization options and other parameters
counter = solutions.ObjectCounter(
    view_img=True,                   # Display the image during processing
    reg_pts=region_points,          # Region of interest points
    names=model.names,  # Class names from the YOLO model
    draw_tracks=True,                # Draw tracking lines for objects
    line_thickness=2,                 # Thickness of the lines drawn
    view_in_counts = True,          # Flag to control whether to display the in counts on the video stream.
    view_out_counts = True        # Flag to control whether to display the out counts on the video stream.
)

# Process video frames in a loop
while cap.isOpened():
    success, im0 = cap.read()
    if not success:
        print("Video frame is empty or video processing has been successfully completed.")
        break

    # Perform object tracking on the current frame, filtering by specified classes
    tracks = model.track(im0, persist=True, show=False, classes=classes_to_count)
    #tracks = model.track(im0, persist=True, show=False)

    # Use the Object Counter to count objects in the frame and get the annotated image
    im0 = counter.start_counting(im0, tracks)

    # Write the annotated frame to the output video
    video_writer.write(im0)

# Release the video capture and writer objects
cap.release()
video_writer.release()

# Close all OpenCV windows
cv2.destroyAllWindows()
