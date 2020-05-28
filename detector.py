import numpy as np
import throttle
import threading
import cv2
import os
from matplotlib import pyplot as plt
import uuid


class Detector:
    def __init__(self,
                 min_confidence=0.5,
                 proto='assets/deploy.prototxt.txt',
                 model='assets/res10_300x300_ssd_iter_140000.caffemodel'):
        self.min_confidence = min_confidence
        self.detected_faces = None
        self.net = cv2.dnn.readNetFromCaffe(proto, model)

    @throttle.wrap(1, 1)
    def process(self, frame):
        print('=> detecting...')
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
        self.net.setInput(blob)
        faces = self.net.forward()

        self.detected_faces = faces
        thread = threading.Thread(target=Detector.save_dataset, args=(frame, faces, self.min_confidence))
        thread.start()

    @staticmethod
    @throttle.wrap(3, 1)
    def save_dataset(frame, faces, min_confidence=0.5, save_in_dir='assets/dataset'):
        (h, w) = frame.shape[:2]
        if not os.path.exists(save_in_dir):
            os.mkdir(save_in_dir)

        for i in range(0, faces.shape[2]):
            confidence = faces[0, 0, i, 2]
            if confidence > min_confidence:
                box = faces[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                image = frame[startY:endY, startX:endX]
                plt.imshow(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
                plt.show()
                cv2.imwrite(save_in_dir+'/'+str(uuid.uuid4())+'.jpeg', image)
