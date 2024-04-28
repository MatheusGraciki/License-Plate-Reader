import cv2

def resizeFrame(frame, scale=0.3):
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)
    dimensions = (width, height)

    return cv2.resize(frame, dimensions, interpolation=cv2.INTER_AREA)


def displayVideo(video_path):
    capture = cv2.VideoCapture(video_path)
    
    if not capture.isOpened():
        print('Error: o vídeo não pode ser aberto.')
        return

    cv2.namedWindow('plates')

    while True:
        frameReadSucess, frame = capture.read()

        if not frameReadSucess:
            print('Erro: A gravação do chegou ao fim')
            break

        resizedFrame = resizeFrame(frame)
        cv2.imshow('highway traffic', resizedFrame)

        # Press "q" to close the window
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    capture.release()
    cv2.destroyAllWindows()


displayVideo("video01.mp4")
