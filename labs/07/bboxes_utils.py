#!/usr/bin/env python3
import numpy as np

from svhn_dataset import SVHN

def bboxes_area(bboxes):
    # Will be implemented by me later; returns area of the given bboxes.
    pass

def bboxes_iou(xs, ys):
    """ Compute IoU for two sets of bboxes xs, ys.

    Each bbox is parametrized as a four-tuple (top, left, bottom, right).
    """
    # Will be implemented by me later; returns area of the given bboxes.
    pass

def bboxes_to_fast_rcnn(anchors, bboxes):
    """ Convert `bboxes` to a Fast-R-CNN-like representation relative to `anchors`.

    The `anchors` and `bboxes` are arrays of four-tuples (top, left, bottom, right);
    you can use SVNH.{TOP, LEFT, BOTTOM, RIGHT} as indices.

    The resulting representation of a single bbox is a four-tuple with:
    - (bbox_y_center - anchor_y_center) / anchor_height
    - (bbox_x_center - anchor_x_center) / anchor_width
    - log(bbox_height / anchor_height)
    - log(bbox_width / anchor_width)
    """

    # TODO: Implement according to the docstring.
    raise NotImplementedError()

def bboxes_from_fast_rcnn(anchor, fast_rcnn):
    """ Convert Fast-R-CNN-like representation relative to `anchor` to a `bbox`."""

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
      is >= threshold, assign the object to the anchor.
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
