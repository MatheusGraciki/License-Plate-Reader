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

The system's capabilities extend to various applications, including:

- Efficient Parking Management: Automating parking fee deductions based on recognized license plates, enhancing convenience for users, and optimizing revenue collection for operators.

- Enhanced Security: Monitoring and tracking vehicles entering and exiting premises, bolstering security measures, and facilitating better control over parking facilities.

## Video Results

[Video Results](https://github.com/MatheusGraciki/License-Plate-Reader/assets/85004422/a3362759-5604-4c3e-b143-7b829f8ef2a1)

## Accessing the Project

This project can be accessed via Google Colab using the following link: [Google Colab Link](https://colab.research.google.com/drive/1hMe21fp9Src31hDu2adCORkAlmzDPdK6?authuser=0#scrollTo=94GeU8KJM4Ar)

## Improving Results with Preprocessing Filters

The project's performance can be enhanced by applying preprocessing filters to the input video frames before license plate detection. These filters may include techniques such as noise reduction, contrast enhancement, and edge detection, improving the accuracy.

## Feedback

We welcome any feedback or suggestions for improvement. Please reach out to us at [dev.matheusgraciki@outlook.com](dev.matheusgraciki@outlook.com).

## Dependencies

- `numpy`
- `supervision`
- `ultralytics`
- `opencv-python`
- `paddleocr`

 





