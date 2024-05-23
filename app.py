import numpy as np
import supervision as sv
from ultralytics import YOLO
import cv2
from paddleocr import PaddleOCR,draw_ocr
# Paddleocr supports Chinese, English, French, German, Korean and Japanese.
# You can set the parameter `lang` as `ch`, `en`, `fr`, `german`, `korean`, `japan`
# to switch the language model in order.
ocr = PaddleOCR(use_angle_cls=True, lang='en', use_gpu=False) # need to run only once to download and load model into memory





# Initialize YOLO model for license plate detection
model = YOLO("drive/MyDrive/Colab/license_plate_detector.pt")

# Initialize tracker and annotators
tracker = sv.ByteTrack(track_activation_threshold=0.45)
box_annotator = sv.BoundingBoxAnnotator()
label_annotator = sv.LabelAnnotator(text_scale=5.0, text_position=sv.Position.TOP_CENTER, text_thickness=20)

# Directory to save cropped image files
output_dir = 'drive/MyDrive/Colab/output'

# Dictionary to store detected text and confidence for each track ID
detected_text_dict = {}
detected_confidence_dict = {}

def crop_image(image: np.ndarray, xyxy: np.ndarray) -> np.ndarray:
    """Crop the region of interest from the image."""
    x1, y1, x2, y2 = np.round(xyxy).astype(int)
    return image[y1:y2, x1:x2]

def apply_ocr(image: np.ndarray) -> (str, float):
    """Apply OCR to the cropped region and return the detected text and confidence."""
    # Apply opening filter to remove noise
    kernel = np.ones((3, 3), np.uint8)
    image = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)

    result = ocr.ocr(image, cls=True)
    detected_text = ""
    detected_confidence = 0.0
    for line in result:
        if line:
            text, confidence = line[0][1]  # Extract detected word
            if text:  # Check if there is text in detection
                detected_text = text
                detected_confidence = confidence
                break

    return detected_text, detected_confidence

def draw_text(image: np.ndarray, text: str, position: tuple, font_scale: float = 5.0, font_thickness: int = 2, font_color: tuple = (0, 0, 255), background_color: tuple = (0, 0, 0)):
    """Draw text with background rectangle on the image."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    # Calculate text size to draw background rectangle
    (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, font_thickness)
    # Define background rectangle coordinates
    background_left = position[0]
    background_top = position[1] - text_height
    background_right = position[0] + text_width
    background_bottom = position[1]
    # Draw background rectangle
    cv2.rectangle(image, (background_left, background_top), (background_right, background_bottom), background_color, -1)
    # Add text over the rectangle
    cv2.putText(image, text, position, font, font_scale, font_color, font_thickness, cv2.LINE_AA)

def process_frame(frame: np.ndarray, frame_number: int) -> np.ndarray:
    """Process each frame of the video."""
    global detected_text_dict
    global detected_confidence_dict

    # Detect objects in the frame using YOLO model for car license plate detection
    results = model(frame)[0]  # Adjust confidence threshold to 0.5
    detections = sv.Detections.from_ultralytics(results)
    detections = tracker.update_with_detections(detections)

    for i, xyxy in enumerate(detections.xyxy):
        # Check if there are tracking IDs
        if detections.tracker_id.size > 0:
            # Get tracking ID of the detection
            track_id = detections.tracker_id[i]
            # Check if the car plate in the image has already been cropped or not, based on the tracking ID
            if track_id not in detected_text_dict:
                # Crop the region of interest (car plate)
                cropped_region = crop_image(frame, xyxy)

                # Apply OCR to the cropped region
                detected_text, detected_confidence = apply_ocr(cropped_region)

                # Store the detected text and confidence in the dictionary for the corresponding track ID
                detected_text_dict[track_id] = detected_text
                detected_confidence_dict[track_id] = detected_confidence

            # Draw a box around the car plate
            frame = box_annotator.annotate(frame, detections=detections)

            # Annotate the frame with OCR result
            if track_id in detected_text_dict:
                detected_text = detected_text_dict[track_id]
                detected_confidence = detected_confidence_dict[track_id]
                if detected_text:
                    # Draw the detected text on the frame with background rectangle
                    label = f"{detected_text}, {detected_confidence:.2f}"
                    draw_text(frame, label, (int(xyxy[0] -400), int(xyxy[1] - 60)), font_scale=5.0, font_thickness=9, font_color=(255, 255, 255), background_color=(0, 0, 0))
                
    return frame

# Process the video with the callback function
sv.process_video(
    source_path="drive/MyDrive/Colab/video02.mp4",
    target_path="drive/MyDrive/Colab/result.mp4",
    callback=process_frame
)
