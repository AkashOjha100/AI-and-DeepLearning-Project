import cv2
import torch
import time
import os

# Load the YOLOv5 model (you can choose 'yolov5s', 'yolov5m', 'yolov5l', or 'yolov5x')
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Initialize video capture (0 for webcam, or provide a video file path)
cap = cv2.VideoCapture(0)

# Create a directory to save detected frames
output_dir = 'detected_frames'
os.makedirs(output_dir, exist_ok=True)

frame_count = 0
start_time = time.time()

while True:
    # Read a frame from the video capture
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame. Exiting...")
        break

    # Perform object detection
    results = model(frame)

    # Render results on the frame
    results.render()  # This modifies the frame in place

    # Calculate FPS
    frame_count += 1
    elapsed_time = time.time() - start_time
    if elapsed_time > 0:
        fps = frame_count / elapsed_time
    else:
        fps = 0

    # Display FPS on the frame
    cv2.putText(frame, f'FPS: {fps:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display the frame with detections
    cv2.imshow('Object Detection', frame)

    # Optionally save the detected frame
    cv2.imwrite(os.path.join(output_dir, f'detected_frame_{frame_count}.jpg'), frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()