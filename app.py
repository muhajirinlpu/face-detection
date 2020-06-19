import os
from threading import Timer
import cv2
from typing import Union
from termcolor import colored
from detector import Detector


class App:
    def __init__(self, path: Union[int, str] = 'video/videoplayback.mp4', min_confidence=0.5):
        self.video_path = path
        self.min_confidence = min_confidence
        self.capture = cv2.VideoCapture(self.video_path)
        self.detector = Detector(detect_identity=True, capture=True)
        if not self.capture.isOpened():
            raise Exception('Error opening video file')

    def start(self, timer=-1):
        while self.capture.isOpened():
            ret, frame = self.capture.read()
            if ret:
                frame = cv2.flip(frame, 1)
                self.detector.process_detected_faces_prop(frame)
                if self.detector.detected_faces is not None:
                    for i in range(0, len(self.detector.detected_faces)):
                        (startX, startY, endX, endY, confidence) = self.detector.detected_faces[i]
                        text = "identifying face | confidence : {:.2f}%".format(confidence * 100)
                        if len(self.detector.detected_faces_identity) - 1 >= i and \
                                self.detector.detected_faces_identity[i] is not None:
                            text = self.detector.detected_faces_identity[i]
                        y = startY - 10 if startY - 10 > 10 else startY + 10
                        cv2.rectangle(frame, (startX - 2, startY - 2), (endX + 2, endY + 2), (0, 0, 255), 2)
                        cv2.putText(frame, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
                if timer > 0:
                    (Timer(timer, self.end)).start()
                cv2.imshow('Frame', frame)
                if cv2.waitKey(25) & 0xFF == ord('q'):
                    self.end()
                    break
            else:
                break

    def end(self):
        self.capture.release()
        cv2.destroyAllWindows()

    def detect_captured(self, capture_dir='assets/capture'):
        print(colored('Detecting picture from assets/capture...', 'green', attrs=['bold']))
        for filename in os.listdir(capture_dir):
            if filename.endswith('.jpeg'):
                image = cv2.imread(capture_dir + '/' + filename)
                print(self.detector.identifier.predict(image))


if __name__ == "__main__":
    video = App(0)
    video.start()
