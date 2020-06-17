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
                 model='assets/res10_300x300_ssd_iter_140000.caffemodel',
                 save_dataset=False):
        self.min_confidence = min_confidence
        # will be typeof None|list<(startX, startY, endX, endY, confidence)>
        self.detected_faces = None
        self.net = cv2.dnn.readNetFromCaffe(proto, model)
        self.save_dataset = save_dataset

    def get_net_raw_faces(self, frame):
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
        self.net.setInput(blob)
        return self.net.forward()

    def get_faces(self, frame):
        faces = self.get_net_raw_faces(frame)

        selected_faces = []
        if faces.shape[2] > 0:
            (h, w) = frame.shape[:2]
            # filter only eligible faces that will be send to property detected_faces and
            # transform type of faces for readable reason
            for i in range(0, faces.shape[2]):
                confidence = faces[0, 0, i, 2]
                if confidence > self.min_confidence:
                    box = faces[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (startX, startY, endX, endY) = box.astype("int")
                    selected_faces.append((startX, startY, endX, endY, confidence))

        return selected_faces

    def get_faces_image(self, frame):
        return list(map(lambda _: Detector.extract_image(frame, _), self.get_faces(frame)))

    @throttle.wrap(5, 1)
    def process_detected_faces_prop(self, frame):
        print('=> detecting...')

        faces = self.get_faces(frame)
        if len(faces) > 0:
            self.detected_faces = faces
            # extract face with start new thread
            if len(faces) > 0:
                thread = threading.Thread(target=Detector.save_dataset, args=(frame, faces))
                thread.start()

    @staticmethod
    def extract_image(frame, face):
        (startX, startY, endX, endY, _) = face
        return frame[startY:endY, startX:endX]

    @staticmethod
    @throttle.wrap(15, 1)
    def save_dataset(frame, faces, save_in_dir='assets/dataset'):
        print('=> extracting faces...')
        if not os.path.exists(save_in_dir):
            os.mkdir(save_in_dir)

        for i in range(0, len(faces)):
            image = Detector.extract_image(frame, faces[i])
            plt.imshow(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            plt.show()
            cv2.imwrite(save_in_dir+'/'+str(uuid.uuid4())+'.jpeg', image)
