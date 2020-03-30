### Assignment: cags_classification
#### Date: Deadline: Apr 12, 23:59
#### Points: 6 points+5 bonus

The goal of this assignment is to use pretrained EfficientNet-B0 model to
achieve best accuracy in CAGS classification.

The [CAGS dataset](https://ufal.mff.cuni.cz/~straka/courses/npfl114/1920/demos/cags_demo.html) consists
of images of **ca**ts and **do**gs of size $224×224$, each classified in one of
the 34 breeds and each containing a mask indicating the presence of the animal.
To load the dataset, use the [cags_dataset.py](https://github.com/ufal/npfl114/tree/master/labs/05/cags_dataset.py)
module. The dataset is stored in a
[TFRecord file](https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset)
and each element is encoded as a
[tf.train.Example](https://www.tensorflow.org/api_docs/python/tf/train/Example).
Therefore the dataset is loaded using `tf.data` API and each entry can be
decoded using `.map(CAGS.parse)` call.
