# License Plate Reader

This project aims to detect vehicle license plates in videos, crop the plate images, extract the text, and display the results with confidence. Additionally, it discusses potential applications of the extracted data for tasks such as vehicle tracking or parking management systems.

## Key Features

- Detection of vehicle license plates and text extraction from videos.
- Real-time display of detections and text overlay on the video.

## How It Works

1. **License Plate Detection**: Utilizes the YOLO (You Only Look Once) model for object detection to identify license plates in video frames.
2. **Text Extraction**: Optical Character Recognition (OCR) is applied to the cropped region of the license plate to extract the text.
3. **Real-time Display**: The video stream is displayed with bounding boxes around detected license plates, and the extracted text is overlaid with confidence scores.

## Potential Applications

The extracted data can be used for various applications, including:

- **Vehicle Tracking**: Analyzing the detected license plates to track the movement of vehicles.
- **Parking Management System**: Implementing a system where registered users can be charged for parking based on detected license plates.

## Feedback

We welcome any feedback or suggestions for improvement. Please reach out to us at [dev.matheusgraciki@outlook.com](dev.matheusgraciki@outlook.com).

## Dependencies

- `numpy`
- `supervision`
- `ultralytics`
- `opencv-python`
- `paddleocr`


