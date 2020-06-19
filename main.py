#!/usr/bin/python

import sys
from detector import Identifier
from app import App
import cv2


def main(argv):
    if 'training' not in argv:
        app = App()

    for action in argv:
        if action == 'training':
            Identifier(Identifier.Treatment.TRAIN)
        elif action == 'capture':
            app.start()
        elif action == 'identify':
            app.detect_captured()
        elif action == 'test':
            i = Identifier()
            i.predict(cv2.imread('./temp/2103181021/8744a10a-19dc-4350-91ca-d88323300d5b.jpeg'))


if __name__ == "__main__":
    main(sys.argv[1:])
