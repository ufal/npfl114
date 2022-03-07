### Assignment: image_augmentation
#### Date: Deadline: Mar 21, 7:59
#### Points: 1 points
#### Tests: image_augmentation_tests

The template [image_augmentation.py](https://github.com/ufal/npfl114/tree/master/labs/04/image_augmentation.py)
creates a simple convolutional network for classifying CIFAR-10.
Your goal is to perform image data augmentation operations using
[ImageDataGenerator](https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/image/ImageDataGenerator)
and to utilize these data during training.

#### Tests Start: image_augmentation_tests
_Note that your results may be slightly different, depending on your CPU type and whether you use a GPU._
- `python3 image_augmentation.py --epochs=1 --batch_size=50`
```
loss: 2.1985 - accuracy: 0.1670 - val_loss: 1.9781 - val_accuracy: 0.2620
```
- `python3 image_augmentation.py --epochs=1 --batch_size=100`
```
loss: 2.1988 - accuracy: 0.1678 - val_loss: 1.9996 - val_accuracy: 0.2680
```
#### Tests End:
