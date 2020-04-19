#!/usr/bin/env python3
import numpy as np

from svhn_dataset import SVHN

def bbox_area(a):
    return max(0, a[SVHN.BOTTOM] - a[SVHN.TOP]) * max(0, a[SVHN.RIGHT] - a[SVHN.LEFT])

def bbox_iou(a, b):
    """ Compute IoU for two bboxes a, b.

    Each bbox is parametrized as a four-tuple (top, left, bottom, right).
    """
    intersection = [
        max(a[SVHN.TOP], b[SVHN.TOP]),
        max(a[SVHN.LEFT], b[SVHN.LEFT]),
        min(a[SVHN.BOTTOM], b[SVHN.BOTTOM]),
        min(a[SVHN.RIGHT], b[SVHN.RIGHT]),
    ]
    if intersection[SVHN.RIGHT] <= intersection[SVHN.LEFT] or intersection[SVHN.BOTTOM] <= intersection[SVHN.TOP]:
        return 0
    return bbox_area(intersection) / float(bbox_area(a) + bbox_area(b) - bbox_area(intersection))

def bbox_to_fast_rcnn(anchor, bbox):
    """ Convert `bbox` to a Fast-R-CNN-like representation relative to `anchor`.

    The `anchor` and `bbox` are four-tuples (top, left, bottom, right);
    you can use SVNH.{TOP, LEFT, BOTTOM, RIGHT} as indices.

    The resulting representation is a four-tuple with:
    - (bbox_y_center - anchor_y_center) / anchor_height
    - (bbox_x_center - anchor_x_center) / anchor_width
    - np.log(bbox_height / anchor_height)
    - np.log(bbox_width / anchor_width)
    """
    assert anchor[SVHN.BOTTOM] > anchor[SVHN.TOP]
    assert anchor[SVHN.RIGHT] > anchor[SVHN.LEFT]
    assert bbox[SVHN.BOTTOM] > bbox[SVHN.TOP]
    assert bbox[SVHN.RIGHT] > bbox[SVHN.LEFT]

    # TODO: Implement according to the docstring.
    raise NotImplementedError()

def bbox_from_fast_rcnn(anchor, fast_rcnn):
    """ Convert Fast-R-CNN-like representation relative to `anchor` to a `bbox`."""
    assert anchor[SVHN.BOTTOM] > anchor[SVHN.TOP]
    assert anchor[SVHN.RIGHT] > anchor[SVHN.LEFT]

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
    - First, gold objects are sequentially processed. For each gold object,
      find the unused anchor with the largest IoU (the first one if there are
      several) and if the IoU is > 0, assign the object to the anchor.
    - Second, anchors unassigned so far are sequentially processed. For each
      anchor, find the gold object with the largest IoU (again the first one if
      there are several), and if the IoU is >= threshold, assign the object to
      the anchor.
    """

    anchor_classes = np.zeros(len(anchors), np.int32)
    anchor_bboxes = np.zeros([len(anchors), 4], np.float32)

    # TODO: Sequentially for each gold object, find the unused anchor
    # with the largest IoU (the first one if there are several)
    # and if the IoU is > 0, assign the object to the anchor.

    # TODO: Sequentially for each unassigned anchor, find the gold object
    # with the largest IoU (the first one if there are several).
    # If the IoU >= threshold, assign the object to the anchor.

    return anchor_classes, anchor_bboxes

import unittest
class Tests(unittest.TestCase):
    def test_bbox_to_from_fast_rcnn(self):
        for anchor, bbox, fast_rcnn in [
                [[0, 0, 10, 10], [0, 0, 10, 10], [0, 0, 0, 0]],
                [[0, 0, 10, 10], [5, 0, 15, 10], [.5, 0, 0, 0]],
                [[0, 0, 10, 10], [0, 5, 10, 15], [0, .5, 0, 0]],
                [[0, 0, 10, 10], [0, 0, 20, 20], [.5, .5, np.log(2), np.log(2)]],
        ]:
            np.testing.assert_almost_equal(bbox_to_fast_rcnn(anchor, bbox), fast_rcnn, decimal=3)
            np.testing.assert_almost_equal(bbox_from_fast_rcnn(anchor, fast_rcnn), bbox, decimal=3)

    def test_bboxes_training(self):
        anchors = [[0, 0, 10, 10], [0, 10, 10, 20], [10, 0, 20, 10], [10, 10, 20, 20]]
        for gold_classes, gold_bboxes, anchor_classes, anchor_bboxes, iou in [
                [[1], [[14, 14, 16, 16]], [0, 0, 0, 2], [[0, 0, 0, 0]] * 3 + [[0, 0, np.log(1/5), np.log(1/5)]], 0.5],
                [[2], [[0, 0, 20, 20]], [3, 0, 0, 0], [[.5, .5, np.log(2), np.log(2)]] + [[0, 0, 0, 0]] * 3, 0.26],
                [[2], [[0, 0, 20, 20]], [3, 3, 3, 3], [[y, x, np.log(2), np.log(2)] for y in [.5, -.5] for x in [.5, -.5]], 0.24],
        ]:
            computed_classes, computed_bboxes = bboxes_training(anchors, gold_classes, gold_bboxes, iou)
            np.testing.assert_almost_equal(computed_classes, anchor_classes, decimal=3)
            np.testing.assert_almost_equal(computed_bboxes, anchor_bboxes, decimal=3)

if __name__ == '__main__':
    unittest.main()
