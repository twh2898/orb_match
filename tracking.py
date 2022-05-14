#!/usr/bin/env python3

class Track:
    def __init__(self, id_):
        self.id = id_
        self.points = []

    def add_point(self, u, v, t):
        self.points.append({'u': u, 'v': v, 't': t})

    def serialize(self):
        return {'id': self.id, 'points': self.points}


class Tracker:
    def __init__(self):
        self.tracks = []
        self.live_tracks = {}
        self.t = 0

    def add_frame(self, matches, train_kp):
        tracks = {}
        live_tracks = self.live_tracks

        for match in matches:
            queryIdx = match[0].queryIdx
            trainIdx = match[0].trainIdx

            if queryIdx in live_tracks:
                track = live_tracks[queryIdx]
            else:
                track = Track(len(self.tracks))
                self.tracks.append(track)

            u, v = train_kp[trainIdx].pt

            track.add_point(u, v, self.t)
            tracks[trainIdx] = track

        self.live_tracks = tracks
        self.t += 1

    def serialize(self, filter_len=None):
        tracks = self.tracks
        if filter_len is not None:
            tracks = list(filter(lambda t: len(t.points)
                          >= filter_len, self.tracks))
        return [t.serialize() for t in tracks]
