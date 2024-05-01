import numpy as np
import supervision as sv
from ultralytics import YOLO
import cv2
import os

# Initialize the model for plate reading, tracking, and drawing a box around the plate.
model = YOLO("drive/MyDrive/Colab/license_plate_detector.pt")
tracker = sv.ByteTrack(track_activation_threshold=0.60)
box_annotator = sv.BoundingBoxAnnotator()

# Directory to save cropped image files.
output_dir = 'drive/MyDrive/Colab/output'

# Set to store unique tracker IDs of cropped plates.
cropped_track_ids = set()

def crop_image(image: np.ndarray, xyxy: np.ndarray) -> np.ndarray:
    """Crops the region of interest from the image."""
    xyxy = np.round(xyxy).astype(int)
    x1, y1, x2, y2 = xyxy
    cropped_img = image[y1:y2, x1:x2]
    return cropped_img

def callback(frame: np.ndarray, frame_number: int) -> np.ndarray:
    """Processes each frame of the video."""
    global cropped_track_ids

    # Detect objects in the frame using the license plate recognition model.
    results = model(frame)[0]
    detections = sv.Detections.from_ultralytics(results)
    detections = tracker.update_with_detections(detections)

    for i, xyxy in enumerate(detections.xyxy):
        # Check if there are tracker IDs.
        if detections.tracker_id.size > 0:
            # Get the tracker ID of the detection.
            track_id = detections.tracker_id[i]
            # Check if the car plate in the image has already been cropped or not, based on the tracking id.
            if track_id not in cropped_track_ids:
                # Crop the region of interest (car plate).
                cropped_region = crop_image(frame, xyxy)

                # Create a unique image name based on the tracking id and the detection index.
                image_name = f"{track_id}_TrackId_detection_{i}.jpg"
          
                # Save the cropped region as an image.
                cv2.imwrite(os.path.join(output_dir, image_name), cropped_region)

                # Add the tracker ID to the set of IDs that have already had the image cropped.
                cropped_track_ids.add(track_id)

    # Draw a box around the car plate.
    return box_annotator.annotate(frame.copy(), detections=detections)

# Process the video with the callback function.
sv.process_video(
    source_path="drive/MyDrive/Colab/video02.mp4",
    target_path="drive/MyDrive/Colab/result.mp4",
    callback=callback
)
