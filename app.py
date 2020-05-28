import cv2
import numpy as np
from typing import Union

from detector import Detector


class App:
    def __init__(self, path: Union[int, str] = 'video/videoplayback.mp4', min_confidence=0.5):
        self.video_path = path
        self.min_confidence = min_confidence
        self.capture = cv2.VideoCapture(self.video_path)
        self.detector = Detector()
        if not self.capture.isOpened():
            raise Exception('Error opening video file')

    def start(self):
        while self.capture.isOpened():
            ret, frame = self.capture.read()
            if ret:
                self.detector.process(frame)
                if self.detector.detected_faces is not None:
                    for i in range(0, len(self.detector.detected_faces)):
                        (startX, startY, endX, endY, confidence) = self.detector.detected_faces[i]
                        text = "{:.2f}%".format(confidence * 100)
                        y = startY - 10 if startY - 10 > 10 else startY + 10
                        cv2.rectangle(frame, (startX - 2, startY - 2), (endX + 2, endY + 2), (0, 0, 255), 2)
                        cv2.putText(frame, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

                cv2.imshow('Frame', frame)
                if cv2.waitKey(25) & 0xFF == ord('q'):
                    self.end()
                    break
            else:
                break

    def end(self):
        self.capture.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    video = App(min_confidence=0.5)
    video.start()
