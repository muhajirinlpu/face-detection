#!/usr/bin/python

import sys
from enum import Enum
from detector import Identifier


def main(argv):
    for action in argv:
        if action == 'train_dataset':
            Identifier()


if __name__ == "__main__":
    main(sys.argv[1:])
