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
            i.predict(cv2.imread('./temp/2103181021/8744a10a-19dc-4350-91ca-d88323300d5b.jpeg'))


if __name__ == "__main__":
    main(sys.argv[1:])
