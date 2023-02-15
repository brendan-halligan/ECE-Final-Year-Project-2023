from __future__      import print_function

import argparse
import glob
import os
import matplotlib
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import time

from filterpy.kalman import KalmanFilter
from skimage         import io


np.random.seed(0)
matplotlib.use('TkAgg')


def linear_assignment(cost_matrix):
    """
    Function takes in a cost matrix and solves the Linear Assignment Problem (LAP).

    :param cost_matrix: Cost matrix to be solved.
    :return: An array of pairs of the form [y[i], i] for i in x, where x and y are the solutions to the LAP.
    """
    """
    Attempt to use the ``lapjv`` function from the ``lap`` library.
    If the ``lap`` library is unavailable, it uses the ``linear_sum_assignment`` function from the ``scipy.optimize``
    library instead.
    """
    try:
        import lap
        _, x, y = lap.lapjv(cost_matrix, extend_cost=True)
        return np.array([[y[i], i] for i in x if i >= 0])
    except ImportError:
        from scipy.optimize import linear_sum_assignment
        x, y = linear_sum_assignment(cost_matrix)
        return np.array(list(zip(x, y)))


def iou_batch(bb_test, bb_gt):
    """
    Function to calculate the Intersection over Union (IoU) between two sets of boundary boxes.

    :param bb_test: Test boundary boxes.
    :param bb_gt:   Ground truth boundary boxes.
    :return:        The calculated IoU.
    """
    bb_gt = np.expand_dims(bb_gt, 0)
    bb_test = np.expand_dims(bb_test, 1)

    """
    Calculate the coordinates of the intersection bounding box by taking the maximum of the x coordinates and 
    y coordinates of the test and ground truth boundary boxes.
    """
    xx1 = np.maximum(bb_test[..., 0], bb_gt[..., 0])
    yy1 = np.maximum(bb_test[..., 1], bb_gt[..., 1])
    xx2 = np.minimum(bb_test[..., 2], bb_gt[..., 2])
    yy2 = np.minimum(bb_test[..., 3], bb_gt[..., 3])

    # Calculate the width and height of the intersection boundary box and then the area of the intersection.
    w  = np.maximum(0., xx2 - xx1)
    h  = np.maximum(0., yy2 - yy1)
    wh = w * h

    # Calculate the IoU by dividing the area of intersection by the total area of both boundary boxes.
    o = wh / ((bb_test[..., 2] - bb_test[..., 0]) * (bb_test[..., 3] - bb_test[..., 1])
              + (bb_gt[..., 2] - bb_gt[..., 0]) * (bb_gt[..., 3] - bb_gt[..., 1]) - wh)
    return o


def convert_bbox_to_z(bbox):
    """
    Converts a boundary box from the form [x1, y1, x2, y2] to the form [x, y, s, r], where:
        1. x1, y1: The top left co-ordinate of the boundary box.
        2. x2, y2: The bottom right co-ordinate of the boundary box
        3. x: X location of the centre of the boundary box.
        4. y: Y location of the centre of the boundary box.
        5. s: Scale/area.
        6. r: Aspect ratio.

    :param bbox: Boundary box in the form [x1, y1, x2, y2].
    :return:     Boundary box in the form [x, y, s, r].
    """
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    x = bbox[0] + w / 2.
    y = bbox[1] + h / 2.
    s = w * h
    r = w / float(h)

    return np.array([x, y, s, r]).reshape((4, 1))


def convert_x_to_bbox(x, score=None):
    """
    Converts a boundary box from the form [x, y, s, r] to the form [x1, y1, x2, y2].
        1. x: X location of the centre of the boundary box.
        2. y: Y location of the centre of the boundary box.
        3. s: Scale/area.
        4. r: Aspect ratio.
        5. x1, y1: The top left co-ordinate of the boundary box.
        6. x2, y2: The bottom right co-ordinate of the boundary box

    :param x:     Boundary box in the form [x, y, s, r].
    :param score: Set to None.
    :return:      Boundary box in the form [x1, y1, x2, y2].
    """
    w = np.sqrt(x[2] * x[3])
    h = x[2] / w

    if score is None:
        return np.array([x[0] - w / 2., x[1] - h / 2., x[0] + w / 2., x[1] + h / 2.]).reshape((1, 4))
    else:
        return np.array([x[0] - w / 2., x[1] - h / 2., x[0] + w / 2., x[1] + h / 2., score]).reshape((1, 5))


class KalmanBoxTracker(object):
    """
    KalmanBoxTracker class represents the internal state of individual tracker objects observed as a boundary box.
    """
    count = 0

    def __init__(self, bbox):
        """
        Initialise a tracker object using an initial boundary box.

        :param bbox: Initial boundary box which the KalmanBoxTracker is based off of. Must have a ``detected class``
                     integer number at the -1 position.
        """
        self.kf   = KalmanFilter(dim_x=7, dim_z=4)
        self.kf.F = np.array(
            [[1, 0, 0, 0, 1, 0, 0], [0, 1, 0, 0, 0, 1, 0], [0, 0, 1, 0, 0, 0, 1], [0, 0, 0, 1, 0, 0, 0],
             [0, 0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 0, 1]])
        self.kf.H = np.array(
            [[1, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0]])

        self.kf.R[2:, 2:] *= 10.    # R: Covariance matrix of measurement noise (set to high for noisy inputs).
        self.kf.P[4:, 4:] *= 1000.  # Give high uncertainty to the unobservable initial velocities.
        self.kf.P         *= 10.
        self.kf.Q[-1, -1] *= 0.5    # Q: Covariance matrix of process noise (set to high for erratically moving things).
        self.kf.Q[4:, 4:] *= 0.5

        # State vector.
        self.kf.x[:4]          = convert_bbox_to_z(bbox)
        self.time_since_update = 0

        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1

        self.history     = []
        self.hits        = 0
        self.hit_streak  = 0
        self.age         = 0
        self.centroidarr = []
        CX               = (bbox[0] + bbox[2]) // 2
        CY               = (bbox[1] + bbox[3]) // 2
        self.centroidarr.append((CX, CY))

        # Keep YOLOv7 detected class information.
        self.detclass = bbox[5]

    def update(self, bbox):
        """
        Update the state vector with the observed box passed as an argument.

        :param bbox: Boundary box which will update the state vector.
        :return:     None.
        """
        self.time_since_update = 0
        self.history           = []
        self.hits             += 1
        self.hit_streak       += 1
        self.kf.update(convert_bbox_to_z(bbox))

        self.detclass = bbox[5]
        CX            = (bbox[0] + bbox[2]) // 2
        CY            = (bbox[1] + bbox[3]) // 2
        self.centroidarr.append((CX, CY))

    def predict(self):
        """
        Advances the state vector and returns an estimate of the predicted boundary box.

        :return: An estimate of the predicted boundary box.
        """
        if (self.kf.x[6] + self.kf.x[2]) <= 0:
            self.kf.x[6] *= 0.0

        self.kf.predict()
        self.age += 1

        if self.time_since_update > 0:
            self.hit_streak = 0

        self.time_since_update += 1
        self.history.append(convert_x_to_bbox(self.kf.x))

        return self.history[-1]

    def get_state(self):
        """
        Returns an estimate of the current boundary box

        :return: Estimate of current boundary box
        """
        arr_detclass = np.expand_dims(np.array([self.detclass]), 0)
        arr_u_dot    = np.expand_dims(self.kf.x[4], 0)
        arr_v_dot    = np.expand_dims(self.kf.x[5], 0)
        arr_s_dot    = np.expand_dims(self.kf.x[6], 0)

        return np.concatenate((convert_x_to_bbox(self.kf.x), arr_detclass, arr_u_dot, arr_v_dot, arr_s_dot), axis=1)


def associate_detections_to_trackers(detections, trackers, iou_threshold=0.3):
    """
    Assigns detections to a tracker object.
    Both the detections and the tracker objects themselves are represented as boundary boxes.

    :param detections:    Detection boundary boxes.
    :param trackers:      Tracker objects to be updated.
    :param iou_threshold: Threshold value, set to 0.3.

    :return: Three lists comprising:
                1. Matches
                2. Unmatched detections
                3. Unmatched trackers
    """
    # Return empty arrays if there are no tracker objects.
    if len(trackers) == 0:
        return np.empty((0, 2), dtype=int), np.arange(len(detections)), np.empty((0, 5), dtype=int)

    iou_matrix = iou_batch(detections, trackers)

    if min(iou_matrix.shape) > 0:
        a = (iou_matrix > iou_threshold).astype(np.int32)
        if a.sum(1).max() == 1 and a.sum(0).max() == 1:
            matched_indices = np.stack(np.where(a), axis=1)
        else:
            matched_indices = linear_assignment(-iou_matrix)
    else:
        matched_indices = np.empty(shape=(0, 2))

    # Append array of unmatched detections.
    unmatched_detections = []
    for d, det in enumerate(detections):
        if d not in matched_indices[:, 0]:
            unmatched_detections.append(d)

    # Append array of unmatched trackers.
    unmatched_trackers = []
    for t, trk in enumerate(trackers):
        if t not in matched_indices[:, 1]:
            unmatched_trackers.append(t)

    # Filter out matched with low IOU.
    matches = []
    for m in matched_indices:
        if iou_matrix[m[0], m[1]] < iou_threshold:
            unmatched_detections.append(m[0])
            unmatched_trackers.append(m[1])
        else:
            matches.append(m.reshape(1, 2))

    # Set array of matches.
    if len(matches) == 0:
        matches = np.empty((0, 2), dtype=int)
    else:
        matches = np.concatenate(matches, axis=0)

    return matches, np.array(unmatched_detections), np.array(unmatched_trackers)


class Sort(object):

    def __init__(self, max_age=1, min_hits=3, iou_threshold=0.3):
        """
        Initialise a SORT object.

        :param max_age:       Set to 1 by default.
        :param min_hits:      Set to 3 by default.
        :param iou_threshold: Threshold set to 0.3 by default.
        """
        self.max_age       = max_age
        self.min_hits      = min_hits
        self.iou_threshold = iou_threshold
        self.trackers      = []
        self.frame_count   = 0

    def getTrackers(self, ):
        """
        Return trackers.

        :return: Array of tracker objects
        """
        return self.trackers

    def update(self, dets=np.empty((0, 6))):
        """
        Update function must be called for every frame, even if it has no detections (in this case pass np.empty((0, 5))
        as the argument.
        NB: The number of objects returned may differ from the number of objects provided.

        :param dets: A numpy array of detections in the format:
                    [[x1, y1, x2, y2, score], [x1, y1, x2, y2, score], ..., [x1, y1, x2, y2, score]
        :return: An array where the last column is the object ID (replacing the confidence score).
        """
        self.frame_count += 1

        # Get predicted locations from existing trackers.
        trks   = np.zeros((len(self.trackers), 6))
        to_del = []
        ret    = []
        for t, trk in enumerate(trks):
            pos    = self.trackers[t].predict()[0]
            trk[:] = [pos[0], pos[1], pos[2], pos[3], 0, 0]
            if np.any(np.isnan(pos)):
                to_del.append(t)
        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))

        for t in reversed(to_del):
            self.trackers.pop(t)
        matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(dets, trks, self.iou_threshold)

        # Update matched trackers with assigned detections.
        for m in matched:
            self.trackers[m[1]].update(dets[m[0], :])

        # Create and initialize new trackers for unmatched detections.
        for i in unmatched_dets:
            trk = KalmanBoxTracker(np.hstack((dets[i, :], np.array([0]))))
            self.trackers.append(trk)

        i = len(self.trackers)
        for trk in reversed(self.trackers):
            d = trk.get_state()[0]
            if (trk.time_since_update < 1) and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
                # +1 as MOT benchmark requires positive value.
                ret.append(np.concatenate((d, [trk.id + 1])).reshape(1, -1))
            i -= 1

            # Remove expired track.
            if trk.time_since_update > self.max_age:
                self.trackers.pop(i)

        if len(ret) > 0:
            return np.concatenate(ret)

        return np.empty((0, 6))


def parse_args():
    """
    Parse input arguments.

    :return: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="SORT demo")
    parser.add_argument("--display",       dest="display",
                        help="Display online tracker output (slow) [False]", action="store_true")
    parser.add_argument("--seq_path",      help="Path to detections.",       type=str, default="data")
    parser.add_argument("--phase",         help="Subdirectory in seq_path.", type=str, default="train")
    parser.add_argument("--max_age",
                        help="Maximum number of frames to keep alive a track without associated detections.",
                        type=int, default=1)
    parser.add_argument("--min_hits",      help="Minimum number of associated detections before track is initialised.",
                        type=int, default=3)
    parser.add_argument("--iou_threshold", help="Minimum IOU for match.", type=float, default=0.3)
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args()
    display = args.display
    phase = args.phase
    total_time = 0.0
    total_frames = 0
    colours = np.random.rand(32, 3)  # Used only for display

    if display:
        if not os.path.exists("mot_benchmark"):
            print(
                "\n\tERROR: mot_benchmark link not found!\n\n    Create a symbolic link to the MOT benchmark\n    ("
                "https://motchallenge.net/data/2D_MOT_2015/#download). E.g.:\n\n    $ ln -s "
                "/path/to/MOT2015_challenge/2DMOT2015 mot_benchmark\n\n")
        exit()

    plt.ion()
    fig = plt.figure()
    ax1 = fig.add_subplot(111, aspect="equal")

    if not os.path.exists("output"):
        os.makedirs("output")

    pattern = os.path.join(args.seq_path, phase, "*", "det", "det.txt")

    # Create several instances of the MOT tracker.
    for seq_dets_fn in glob.glob(pattern):
        mot_tracker = Sort(max_age=args.max_age, min_hits=args.min_hits, iou_threshold=args.iou_threshold)

    seq_dets = np.loadtxt(seq_dets_fn, delimiter=',')
    seq = seq_dets_fn[pattern.find("*"):].split(os.path.sep)[0]

    with open(os.path.join("output", "%s.txt" % seq), "w") as out_file:
        print("Processing %s." % seq)
        for frame in range(int(seq_dets[:, 0].max())):
            # Detection and frame numbers begin at 1.
            frame += 1
            dets = seq_dets[seq_dets[:, 0] == frame, 2:7]
            # Convert to [x1, y1, w, h] to [x1, y1, x2, y2].
            dets[:, 2:4] += dets[:, 0:2]
            total_frames += 1

        if display:
            fn = os.path.join("mot_benchmark", phase, seq, "img1", "%06d.jpg" % frame)
            im = io.imread(fn)
            ax1.imshow(im)
            plt.title(seq + " Tracked Targets")

        start_time  = time.time()
        trackers    = mot_tracker.update(dets)
        cycle_time  = time.time() - start_time
        total_time += cycle_time

        for d in trackers:
            print("%d,%d,%.2f,%.2f,%.2f,%.2f,1,-1,-1,-1" % (frame, d[4], d[0], d[1], d[2] - d[0], d[3] - d[1]),
                  file=out_file)
            if display:
                d = d.astype(np.int32)
                ax1.add_patch(patches.Rectangle((d[0], d[1]), d[2] - d[0], d[3] - d[1], fill=False, lw=3,
                                                ec=colours[d[4] % 32, :]))

        if display:
            fig.canvas.flush_events()
            plt.draw()
            ax1.cla()

    print("Total Tracking took: %.3f seconds for %d frames or %.1f FPS" % (
        total_time, total_frames, total_frames / total_time))

    if display:
        print("Note: to get real runtime results run without the option: --display")
