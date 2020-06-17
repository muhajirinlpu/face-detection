import numpy as np
import throttle
import threading
import cv2
import os
from enum import Enum
from matplotlib import pyplot as plt
import tempfile
import tensorflow as tf
from tensorflow.keras.models import Sequential, model_from_json
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
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
            cv2.imwrite(save_in_dir + '/' + str(uuid.uuid4()) + '.jpeg', image)


class Identifier:
    class Treatment(Enum):
        TRAIN = 'train'
        LOAD = 'load'

    def __init__(self,
                 treatment=Treatment.LOAD,
                 dataset_dir='./assets/known_dataset',
                 dataset_location='./assets/known_data/dataset.json'):
        self.model = None
        self.dataset_dir = dataset_dir
        self.dataset_location = dataset_location
        if not os.path.exists(dataset_location):
            treatment = Identifier.Treatment.TRAIN

        if treatment == Identifier.Treatment.TRAIN:
            self.train()
        else:
            self.load()

    def prepare_dataset(self, tempdir):
        mapper = {}
        for dir_path, dir_names, files in os.walk(self.dataset_dir):
            print(f'Found directory: {dir_path}, {os.path.basename(dir_path)}')
            # for file_name in files:

        pass

    @staticmethod
    def create_model():
        model = Sequential()
        model.add(Conv2D(16, (3, 3), activation='relu', input_shape=(150, 150, 3)))
        model.add(MaxPooling2D(2, 2))
        model.add(Conv2D(32, (3, 3), activation='relu'))
        model.add(MaxPooling2D(2, 2))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D(2, 2))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D(2, 2))
        model.add(Flatten())
        model.add(Dense(512, activation='relu'))
        model.add(Dense(26, activation='softmax'))

        model.compile(loss='categorical_crossentropy',
                      optimizer=tf.optimizers.Adam(),
                      metrics=['accuracy'])

        return model

    def train(self):
        print('=> Start training...')
        with tempfile.TemporaryDirectory('dataset') as tempdir:
            # self.prepare_dataset(tempdir)

            datagen = ImageDataGenerator(rescale=1. / 255,
                                         shear_range=0.2,
                                         validation_split=0.2,
                                         horizontal_flip=True)

            train_generator = datagen.flow_from_directory(self.dataset_dir,
                                                          target_size=(150, 150),
                                                          batch_size=4,
                                                          class_mode='categorical',
                                                          subset="training")

            validation_generator = datagen.flow_from_directory(self.dataset_dir,
                                                               target_size=(150, 150),
                                                               batch_size=4,
                                                               class_mode='categorical',
                                                               subset="validation")

            model = Identifier.create_model()
            model.fit(train_generator,
                      steps_per_epoch=30,
                      epochs=20,
                      validation_data=validation_generator,
                      validation_steps=4,
                      verbose=2)

            json_model = model.to_json()
            with open(self.dataset_location, 'w') as json_file:
                json_file.write(json_model)
                model.save_weights(self.dataset_location + '.h5')

            model.summary()

        return model

    def load(self):
        with open(self.dataset_location, 'r') as json_file:
            json_saved_model = json_file.read()

        model = model_from_json(json_saved_model)
        model.load_weights(self.dataset_location + '.h5')
        model.compile(loss='categorical_crossentropy',
                      optimizer=tf.optimizers.Adam(),
                      metrics=['accuracy'])

        model.summary()

        return model
