#!/usr/bin/python

import sys
from detector import Identifier
import cv2


def main(argv):
    for action in argv:
        if action == 'train_dataset':
            Identifier(Identifier.Treatment.TRAIN)
        elif action == 'test':
            i = Identifier()
            i.predict(cv2.imread('./assets/known_dataset/2103181004/2103181004 (2).jpeg'))


if __name__ == "__main__":
    main(sys.argv[1:])
