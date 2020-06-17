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
            i.predict(cv2.cvtColor(cv2.imread('./assets/known_dataset/2103181006/2103181006 (2).jpeg'),
                                   cv2.COLOR_RGB2BGR))


if __name__ == "__main__":
    main(sys.argv[1:])
