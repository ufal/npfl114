### Assignment: bboxes_utils
#### Date: Deadline: Apr 19, 23:59
#### Points: 2 points

This is a preparatory assignment for `svhn_competition`. The goal is to
implement several bounding box manipulation routines in the
[bboxes_utils.py](https://github.com/ufal/npfl114/tree/master/labs/06/bboxes_utils.py)
module. Notably, you need to implement the following methods:
- `bbox_to_fast_rcnn`: convert a bounding box to a Fast R-CNN-like
  representation relative to a given anchor;
- `bbox_from_fast_rcnn`: convert a Fast R-CNN-like representation relative to an
  anchor back to a bounding box;
- `bboxes_training`: given a list of anchors and gold objects, assign gold
  objects to anchors and generate suitable training data (the exact algorithm
  is described in the template).

The [bboxes_utils.py](https://github.com/ufal/npfl114/tree/master/labs/06/bboxes_utils.py)
contains simple unit tests, which are evaluated when executing the module,
which you can use to check the validity of your implementation.

When submitting to ReCodEx, you must submit exactly one Python source with
methods `bbox_to_fast_rcnn`, `bbox_to_fast_rcnn` and `bboxes_training`.
These methods are then executed and compared to the reference implementation.
