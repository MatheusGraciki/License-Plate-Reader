import cv2
def displayVideo(video_path):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print('Error: the video cannot be opened.')
        return

    cv2.namedWindow('camera capture')

    while True:
        ret, frame = cap.read()

        if not ret:
            print('Failed to read the frame')
            break

        cv2.imshow('highway traffic', frame)

        # Press "q" to exit the window
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


displayVideo("video.mov")
