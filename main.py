#!/usr/bin/env python3

import cv2
from capture import get_video
from tracking import Tracker


orb = cv2.ORB_create()
matcher = cv2.BFMatcher()

# FLANN parameters
FLANN_INDEX_LSH = 6
index_params = dict(algorithm=FLANN_INDEX_LSH,
                    table_number=6,  # 12
                    key_size=12,     # 20
                    multi_probe_level=1)  # 2
search_params = dict(checks=50)   # or pass empty dictionary

flann = cv2.FlannBasedMatcher(index_params, search_params)


def get_video_with_data(use_bw=False):
    frame = get_video()
    if use_bw:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    kp, des = orb.detectAndCompute(frame, None)
    return frame, kp, des


def ratio_test(matches, ratio=0.75):
    good = []
    for match in matches:
        if len(match) != 2:
            continue

        m, n = match
        if m.distance < ratio * n.distance:
            good.append([m])

    return good


if __name__ == '__main__':
    from statistics import mean, stdev

    last_frame, last_frame_kp, last_frame_des = get_video_with_data()

    tracker = Tracker()

    # 40 is ~2 seconds of data
    for _ in range(40):
        frame, frame_kp, frame_des = get_video_with_data()

        if frame_des is None:
            print('No matches')
            continue

        # matches = matcher.match(last_frame_des, frame_des)
        matches = matcher.knnMatch(last_frame_des, frame_des, k=2)
        # matches = flann.knnMatch(last_frame_des, frame_des, k=2)

        # Apply ratio test
        matches = ratio_test(matches)

        tracker.add_frame(matches, frame_kp)

        # matches = sorted(matches, key=lambda x: x.distance)

        final_img = cv2.drawMatchesKnn(last_frame, last_frame_kp,
                                       frame, frame_kp, matches, None)

        # final_img = cv2.resize(final_img, (1000, 650))
        cv2.imshow('final', final_img)
        if cv2.waitKey(10) == 27:
            break

        last_frame, last_frame_kp, last_frame_des = frame, frame_kp, frame_des

    print('Stats')
    track_len = [len(t.points) for t in tracker.tracks]
    print('N Tracks', len(track_len))
    print('Min', min(track_len), 'Max', max(track_len))
    print('Avg', mean(track_len), '+-', stdev(track_len))

    def n_above(n):
        above = filter(lambda t: t >= n, track_len)
        print('N >=', n, 'is', len(list(above)))

    n_above(30)
    n_above(20)
    n_above(10)

    print('Dumping tracks')
    import yaml
    with open('tracks.yaml', 'w') as f:
        data = tracker.serialize(filter_len=20)
        yaml.safe_dump(data, f)
