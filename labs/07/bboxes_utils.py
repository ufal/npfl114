#!/usr/bin/env python3
import numpy as np

BACKEND = np # or you can use `tf` for TensorFlow implementation

TOP, LEFT, BOTTOM, RIGHT = range(4)

def bboxes_area(bboxes):
    """ Compute area of given set of bboxes.

    The computation can be performed either using Numpy or TensorFlow.
    Each bbox is parametrized as a four-tuple (top, left, bottom, right).

    If the bboxes.shape is [..., 4], the output shape is bboxes.shape[:-1].
    """
    return BACKEND.maximum(bboxes[..., BOTTOM] - bboxes[..., TOP], 0) \
        * BACKEND.maximum(bboxes[..., RIGHT] - bboxes[..., LEFT], 0)

def bboxes_iou(xs, ys):
    """ Compute IoU of corresponding pairs from two sets of bboxes xs and ys.

    The computation can be performed either using Numpy or TensorFlow.
    Each bbox is parametrized as a four-tuple (top, left, bottom, right).

    Note that broadcasting is supported, so passing inputs with
    xs.shape=[num_xs, 1, 4] and ys.shape=[1, num_ys, 4] will produce output
    with shape [num_xs, num_ys], computing IoU for all pairs of bboxes from
    xs and ys. Formally, the output shape is np.broadcast(xs, ys).shape[:-1].
    """
    intersections = BACKEND.stack([
        BACKEND.maximum(xs[..., TOP], ys[..., TOP]),
        BACKEND.maximum(xs[..., LEFT], ys[..., LEFT]),
        BACKEND.minimum(xs[..., BOTTOM], ys[..., BOTTOM]),
        BACKEND.minimum(xs[..., RIGHT], ys[..., RIGHT]),
    ], axis=-1)

    xs_area, ys_area, intersections_area = bboxes_area(xs), bboxes_area(ys), bboxes_area(intersections)

    return intersections_area / (xs_area + ys_area - intersections_area)

def bboxes_to_fast_rcnn(anchors, bboxes):
    """ Convert `bboxes` to a Fast-R-CNN-like representation relative to `anchors`.

    The `anchors` and `bboxes` are arrays of four-tuples (top, left, bottom, right);
    you can use the TOP, LEFT, BOTTOM, RIGHT constants as indices of the
    respective coordinates.

    The resulting representation of a single bbox is a four-tuple with:
    - (bbox_y_center - anchor_y_center) / anchor_height
    - (bbox_x_center - anchor_x_center) / anchor_width
    - log(bbox_height / anchor_height)
    - log(bbox_width / anchor_width)

    If the anchors.shape is [anchors_len, 4], bboxes.shape is [anchors_len, 4],
    the output shape is [anchors_len, 4].
    """

    # TODO: Implement according to the docstring.
    raise NotImplementedError()

def bboxes_from_fast_rcnn(anchors, fast_rcnns):
    """ Convert Fast-R-CNN-like representation relative to `anchor` to a `bbox`.

    The anchors.shape is [anchors_len, 4], fast_rcnns.shape is [anchors_len, 4],
    the output shape is [anchors_len, 4].
    """

    # TODO: Implement according to the docstring.
    raise NotImplementedError()

def bboxes_training(anchors, gold_classes, gold_bboxes, iou_threshold):
    """ Compute training data for object detection.

    Arguments:
    - `anchors` is an array of four-tuples (top, left, bottom, right)
    - `gold_classes` is an array of zero-based classes of the gold objects
    - `gold_bboxes` is an array of four-tuples (top, left, bottom, right)
      of the gold objects
    - `iou_threshold` is a given threshold

    Returns:
    - `anchor_classes` contains for every anchor either 0 for background
      (if no gold object is assigned) or `1 + gold_class` if a gold object
      with `gold_class` is assigned to it
    - `anchor_bboxes` contains for every anchor a four-tuple
      `(center_y, center_x, height, width)` representing the gold bbox of
      a chosen object using parametrization of Fast R-CNN; zeros if no
      gold object was assigned to the anchor

    Algorithm:
    - First, for each gold object, assign it to an anchor with the largest IoU
      (the one with smaller index if there are several). In case several gold
      objects are assigned to a single anchor, use the gold object with smaller
      index.
    - For each unused anchors, find the gold object with the largest IoU
      (again the one with smaller index if there are several), and if the IoU
      is >= iou_threshold, assign the object to the anchor.
    """

    # TODO: First, for each gold object, assign it to an anchor with the
    # largest IoU (the one with smaller index if there are several). In case
    # several gold objects are assigned to a single anchor, use the gold object
    # with smaller index.

    # TODO: For each unused anchors, find the gold object with the largest IoU
    # (again the one with smaller index if there are several), and if the IoU
    # is >= threshold, assign the object to the anchor.

    return anchor_classes, anchor_bboxes

def main(args):
    return bboxes_to_fast_rcnn, bboxes_from_fast_rcnn, bboxes_training

import unittest
class Tests(unittest.TestCase):
    def test_bboxes_to_from_fast_rcnn(self):
        for anchor, bbox, fast_rcnn in [
                [[[0, 0, 10, 10]], [[0, 0, 10, 10]], [[0,  0, 0, 0]]],
                [[[0, 0, 10, 10]], [[5, 0, 15, 10]], [[.5, 0, 0, 0]]],
                [[[0, 0, 10, 10]], [[0, 5, 10, 15]], [[0, .5, 0, 0]]],
                [[[0, 0, 10, 10]], [[0, 0, 20, 30]], [[.5, 1, np.log(2), np.log(3)]]],
                [[[0, 9, 10, 19]], [[2, 10, 5, 16]], [[-0.15, -0.1, -1.2039728, -0.5108256]]],
                [[[5, 3, 15, 13]], [[7, 7, 10, 9]], [[-0.15, 0, -1.2039728, -1.609438]]],
                [[[7, 6, 17, 16]], [[9, 10, 12, 13]], [[-0.15, 0.05, -1.2039728, -1.2039728]]],
                [[[5, 6, 15, 16]], [[7, 7, 10, 10]], [[-0.15, -0.25, -1.2039728, -1.2039728]]],
                [[[6, 3, 16, 13]], [[8, 5, 12, 8]], [[-0.1, -0.15, -0.9162907, -1.2039728]]],
                [[[5, 2, 15, 12]], [[9, 6, 12, 8]], [[0.05, 0, -1.2039728, -1.609438]]],
                [[[2, 10, 12, 20]], [[6, 11, 8, 17]], [[0, -0.1, -1.609438, -0.5108256]]],
                [[[10, 9, 20, 19]], [[12, 13, 17, 16]], [[-0.05, 0.05, -0.6931472, -1.2039728]]],
                [[[6, 7, 16, 17]], [[10, 11, 12, 14]], [[0, 0.05, -1.609438, -1.2039728]]],
                [[[2, 2, 12, 12]], [[3, 5, 8, 8]], [[-0.15, -0.05, -0.6931472, -1.2039728]]],
        ]:
            anchor, bbox, fast_rcnn = np.array(anchor, np.float32), np.array(bbox, np.float32), np.array(fast_rcnn, np.float32)
            np.testing.assert_almost_equal(bboxes_to_fast_rcnn(anchor, bbox), fast_rcnn, decimal=3)
            np.testing.assert_almost_equal(bboxes_from_fast_rcnn(anchor, fast_rcnn), bbox, decimal=3)

    def test_bboxes_training(self):
        anchors = np.array([[0, 0, 10, 10], [0, 10, 10, 20], [10, 0, 20, 10], [10, 10, 20, 20]], np.float32)
        for gold_classes, gold_bboxes, anchor_classes, anchor_bboxes, iou in [
                [[1], [[14., 14, 16, 16]], [0, 0, 0, 2], [[0, 0, 0, 0]] * 3 + [[0, 0, np.log(1/5), np.log(1/5)]], 0.5],
                [[2], [[0., 0, 20, 20]], [3, 0, 0, 0], [[.5, .5, np.log(2), np.log(2)]] + [[0, 0, 0, 0]] * 3, 0.26],
                [[2], [[0., 0, 20, 20]], [3, 3, 3, 3], [[y, x, np.log(2), np.log(2)] for y in [.5, -.5] for x in [.5, -.5]], 0.24],
                [[0, 1], [[3, 3, 20, 18], [10, 1, 18, 21]], [0, 0, 0, 1],
                 [[0, 0, 0 ,0], [0, 0, 0, 0], [0, 0, 0, 0], [-0.35, -0.45, 0.53062826, 0.4054651]], 0.5],
                [[0, 1], [[3, 3, 20, 18], [10, 1, 18, 21]], [0, 0, 2, 1],
                 [[0, 0, 0 ,0], [0, 0, 0, 0], [-0.1, 0.6, -0.22314353, 0.6931472], [-0.35, -0.45, 0.53062826, 0.4054651]], 0.3],
                [[0, 1], [[3, 3, 20, 18], [10, 1, 18, 21]], [0, 1, 2, 1],
                 [[0, 0, 0 ,0], [0.65, -0.45, 0.53062826, 0.4054651], [-0.1, 0.6, -0.22314353, 0.6931472], [-0.35, -0.45, 0.53062826, 0.4054651]], 0.17],
        ]:
            gold_classes, anchor_classes = np.array(gold_classes, np.int32), np.array(anchor_classes, np.int32)
            gold_bboxes, anchor_bboxes = np.array(gold_bboxes, np.float32), np.array(anchor_bboxes, np.float32)
            computed_classes, computed_bboxes = bboxes_training(anchors, gold_classes, gold_bboxes, iou)
            np.testing.assert_almost_equal(computed_classes, anchor_classes, decimal=3)
            np.testing.assert_almost_equal(computed_bboxes, anchor_bboxes, decimal=3)

if __name__ == '__main__':
    unittest.main()
