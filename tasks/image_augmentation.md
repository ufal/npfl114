### Assignment: image_augmentation
#### Date: Deadline: Apr 05, 23:59
#### Points: 1 points
#### Examples: image_augmentation_examples

The template [image_augmentation.py](https://github.com/ufal/npfl114/tree/master/labs/04/image_augmentation.py)
creates a simple convolutional network for classifying CIFAR-10.
Your goal is to perform image data augmentation
operations using
[ImageDataGenerator](https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/image/ImageDataGenerator)
and to utilize these data during training.
#### Examples Start: image_augmentation_examples
_Note that your results may be slightly different, depending on your CPU type and whether you use GPU._
- `python3 image_augmentation.py --batch_size=50`
```
Epoch 1/5 loss: 2.2698 - accuracy: 0.1253 - val_loss: 1.9850 - val_accuracy: 0.2590
Epoch 2/5 loss: 2.0054 - accuracy: 0.2387 - val_loss: 1.7783 - val_accuracy: 0.3250
Epoch 3/5 loss: 1.8557 - accuracy: 0.3121 - val_loss: 1.7411 - val_accuracy: 0.3620
Epoch 4/5 loss: 1.7431 - accuracy: 0.3565 - val_loss: 1.6151 - val_accuracy: 0.4160
Epoch 5/5 loss: 1.6636 - accuracy: 0.3849 - val_loss: 1.6074 - val_accuracy: 0.4230
```
- `python3 image_augmentation.py --batch_size=100`
```
Epoch 1/5 loss: 2.2671 - accuracy: 0.1350 - val_loss: 1.9996 - val_accuracy: 0.2680
Epoch 2/5 loss: 1.9756 - accuracy: 0.2813 - val_loss: 1.7990 - val_accuracy: 0.3400
Epoch 3/5 loss: 1.8361 - accuracy: 0.3266 - val_loss: 1.6944 - val_accuracy: 0.3550
Epoch 4/5 loss: 1.7677 - accuracy: 0.3546 - val_loss: 1.6714 - val_accuracy: 0.3850
Epoch 5/5 loss: 1.6904 - accuracy: 0.3673 - val_loss: 1.6651 - val_accuracy: 0.3870
```
#### Examples End:
