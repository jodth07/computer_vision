import numpy as np
import cv2 as cv

color = (0, 255, 0)  # Green color in BGR
line_width = 3
radius = 100
point = (0,0)
frame_resized = None
pressed = True

def click(event, x, y, flags, param):
    global point, frame_resized, pressed, color, radius, line_width

    if event == cv.EVENT_LBUTTONDOWN:
        point = (x, y)
        pressed = True
        cv.circle(frame_resized, point, radius, color, line_width)
        print(f"Clicked at: {point}")
    if event == cv.EVENT_LBUTTONUP:
        # pressed = False
        pass
    if event == cv.EVENT_MOUSEMOVE:
        if pressed:
            cv.circle(frame_resized, point, radius, color, line_width)
            print(f"Moved to: {point}")

if __name__ == '__main__':
    cap = cv.VideoCapture(0)
    cv.namedWindow("frame", cv.WINDOW_AUTOSIZE)
    cv.setMouseCallback("frame", click)

    while True:
        ret, frame = cap.read()
        frame_resized = cv.resize(frame, (0, 0), fx=0.5, fy=0.5)

        cv.circle(frame_resized, point, radius, color, line_width)

        cv.imshow('frame', frame_resized)

        waitKey = cv.waitKey(1) & 0xFF
        if waitKey == ord('q'):
            break
        if waitKey == ord('g'):
            color = (0, 0, 255)
        if waitKey == ord('r'):
            color = (0, 255, 0)
        if waitKey == ord('b'):
            color = (255, 0, 0)

    cap.release()
    cv.destroyAllWindows()
