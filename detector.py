import tempfile

import numpy as np
import throttle
import threading
import cv2
import os
from enum import Enum

from tensorflow.python.keras.constraints import maxnorm
from tensorflow.python.keras.layers import AveragePooling2D, Dropout
from termcolor import colored
import tensorflow as tf
from tensorflow.keras.models import Sequential, model_from_json
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import uuid
import json


def list_files(start_path):
    for root, dirs, files in os.walk(start_path):
        level = root.replace(start_path, '').count(os.sep)
        indent = ' ' * 4 * level
        print('{}{}/'.format(indent, os.path.basename(root)))
        sub_indent = ' ' * 4 * (level + 1)
        for f in files:
            print('{}{}'.format(sub_indent, f))


class Detector:
    def __init__(self,
                 min_confidence=0.5,
                 proto='./assets/opencv_face_detector_uint8.pb',
                 model='./assets/opencv_face_detector.pbtxt',
                 detect_identity=False,
                 capture=False):
        self.min_confidence = min_confidence
        self.capture = capture
        self.detect_identity = detect_identity
        # will be typeof None|list<(startX, startY, endX, endY, confidence)>
        self.detected_faces = None
        self.detected_faces_identity = None
        self.net = cv2.dnn.readNetFromTensorflow(proto, model)
        if detect_identity:
            self.identifier = Identifier()

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

    @staticmethod
    def extract_image(frame, face):
        (startX, startY, endX, endY, _) = face
        image = frame[startY:endY, startX:endX]
        shape = np.shape(image)
        if shape[0] == 0 or shape[1] == 0:
            return None
        return image

    @staticmethod
    def get_faces_image(frame, faces=None):
        return list(filter(lambda _: _ is not None,
                           list(map(lambda _: Detector.extract_image(frame, _), faces))))

    @throttle.wrap(1, 5)
    def process_detected_faces_prop(self, frame):
        print('=> detecting...')

        faces = self.get_faces(frame)
        if len(faces) > 0:
            self.detected_faces = faces

            # find identity from capture
            if self.detect_identity:
                (threading.Thread(target=self.find_identities, args=[frame])).start()

            # extract face with start new thread
            if self.capture:
                (threading.Thread(target=Detector.save_dataset_throttled, args=(frame, faces))).start()

    def find_identities(self, frame):
        if self.detected_faces is not None:
            self.detected_faces_identity = []
            for face in self.detected_faces:
                self.detected_faces_identity.append(self.identifier.predict(Detector.extract_image(frame, face)))

    @staticmethod
    @throttle.wrap(5, 1)
    def save_dataset_throttled(frame, faces):
        Detector.save_dataset(frame, faces)

    @staticmethod
    def save_dataset(frame, faces, save_in_dir='assets/capture'):
        print(colored(f'=> extracting {len(faces)} faces...', 'green', attrs=['bold']))
        if not os.path.exists(save_in_dir):
            os.mkdir(save_in_dir)
        transform = Detector.get_faces_image(frame, faces)
        for image in transform:
            cv2.imwrite(save_in_dir + '/' + str(uuid.uuid4()) + '.jpeg', image)


class Identifier:
    class Treatment(Enum):
        TRAIN = 'train'
        LOAD = 'load'

    def __init__(self,
                 treatment=Treatment.LOAD,
                 dataset_dir='./assets/known_dataset',
                 dataset_location='./assets/known_data/capture.json'):
        self.model = None
        self.faces_list = None
        self.dataset_dir = dataset_dir
        self.dataset_location = dataset_location
        if not os.path.exists(dataset_location):
            treatment = Identifier.Treatment.TRAIN

        if treatment == Identifier.Treatment.TRAIN:
            with tempfile.TemporaryDirectory() as tempdir:
                self.train(tempdir)
            # self.train('./temp')
        else:
            self.load()

    @staticmethod
    def seattle_model(input_width=32, input_height=32, feature_maps=32, feature_window_size=(5, 5), dropout1=0.2,
                      dense=128, dropout2=0.5, use_max_pooling=True, pool_size=(2, 2), optimizer='rmsprop'):
        model = Sequential()

        # - 20 feature maps (each feature map is a reduced-size convolution that detects a different feature)
        # - 3 pixel square window
        model.add(Conv2D(feature_maps,
                         feature_window_size,
                         input_shape=(input_width, input_height, 6),
                         padding='same',
                         data_format='channels_last',
                         activation='relu'))

        # - 40 feature maps (add more features)
        # - 3 pixel square window
        model.add(Conv2D(feature_maps,
                         feature_window_size,
                         padding='same',
                         data_format='channels_last',
                         activation='relu'))

        # Pooling layer
        if use_max_pooling:
            model.add(MaxPooling2D(pool_size=pool_size,
                                   data_format='channels_last'))
        else:
            model.add(AveragePooling2D(pool_size=pool_size,
                                       data_format='channels_last'))

        model.add(Dropout(0.2))

        model.add(Flatten())
        model.add(Dense(128,
                        activation='relu',
                        kernel_constraint=maxnorm(3)))

        # Dropout set to 50%.
        model.add(Dropout(0.5))
        model.add(Dense(1, activation='sigmoid'))

        model.compile(loss='categorical_crossentropy',
                      metrics=['binary_accuracy'],
                      optimizer='rmsprop')

        return model

    @staticmethod
    def create_model(length=25):
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
        model.add(Dense(length, activation='softmax'))

        model.compile(loss='categorical_crossentropy',
                      optimizer=tf.optimizers.Adam(),
                      metrics=['accuracy'])

        return model

    def prepare_dataset(self, tempdir):
        faces_list = []
        for dir_path, dir_names, files in os.walk(self.dataset_dir):
            if len(files) == 0:
                continue

            faces_list.append(os.path.basename(dir_path))
            detector = Detector(0.99)
            basename = os.path.basename(dir_path)
            save_in_dir = tempdir + '/' + basename

            print('====> enter folder : ' + basename)
            for file_name in files:
                print('==> load file ' + file_name)
                frame = cv2.imread(dir_path + '/' + file_name)
                if frame is not None:
                    faces = detector.get_faces(frame)
                    Detector.save_dataset(frame, faces, save_in_dir)

        return np.sort(faces_list).tolist()

    def train(self, tempdir='./temp'):
        print('=> Start training...')
        dataset = self.prepare_dataset(tempdir)
        list_files(tempdir)
        datagen = ImageDataGenerator(rescale=1. / 255,
                                     shear_range=0.2,
                                     validation_split=0.2,
                                     horizontal_flip=True)

        train_generator = datagen.flow_from_directory(tempdir,
                                                      target_size=(150, 150),
                                                      batch_size=4,
                                                      class_mode='categorical',
                                                      subset="training")

        validation_generator = datagen.flow_from_directory(tempdir,
                                                           target_size=(150, 150),
                                                           batch_size=4,
                                                           class_mode='categorical',
                                                           subset="validation")
        model = Identifier.create_model(len(dataset))
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
            self.model = model

        with open(os.path.dirname(self.dataset_location) + '/' + 'ref_name.json', 'w') as json_file:
            json_file.write(json.dumps(dataset))
            self.faces_list = dataset

        model.summary()

    def load(self):
        with open(self.dataset_location, 'r') as json_file:
            json_saved_model = json_file.read()

        with open(os.path.dirname(self.dataset_location) + '/' + 'ref_name.json', 'r') as json_file:
            self.faces_list = json.loads(json_file.read())

        model = model_from_json(json_saved_model)
        model.load_weights(self.dataset_location + '.h5')
        model.compile(loss='categorical_crossentropy',
                      optimizer=tf.optimizers.Adam(),
                      metrics=['accuracy'])

        model.summary()
        self.model = model

    def predict(self, img):
        image = np.expand_dims(cv2.cvtColor(cv2.resize(img, (150, 150)), cv2.COLOR_RGB2BGR), axis=0)
        classes = self.model.predict(image, batch_size=4)

        return self.faces_list[np.argmax(classes)]
