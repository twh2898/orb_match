#!/usr/bin/env python3

import socket
import json
from time import time

sock = socket.socket(socket.AF_INET,  # Internet
                     socket.SOCK_DGRAM)  # UDP

start_time = time()


def get_time():
    return time() - start_time


def send(data):
    if not 'timestamp' in data:
        data["timestamp"] = get_time()
    sock.sendto(json.dumps(data).encode(), ("127.0.0.1", 9870))


if __name__ == '__main__':
    from time import sleep
    from math import sin, cos

    while 1:
        sleep(0.05)
        now = get_time()

        send({
            'timestamp': now,
            'sin': sin(now),
            'cos': cos(now),
        })
