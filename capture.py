#!/usr/bin/env python

import freenect
import cv2
import frame_convert2


def get_depth():
    return frame_convert2.pretty_depth_cv(freenect.sync_get_depth()[0])


def get_video():
    return frame_convert2.video_cv(freenect.sync_get_video()[0])


if __name__ == '__main__':
    import sys

    if len(sys.argv) < 2:
        print('Usage:', sys.argv[0], '<sequence number>')
        exit(1)

    sequence = sys.argv[1]

    rgb = get_video()
    d = get_depth()

    cv2.imwrite(sequence + '.ppm', rgb)
    cv2.imwrite(sequence + '.pgm', d)
