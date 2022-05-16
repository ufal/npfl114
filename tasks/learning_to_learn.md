### Assignment: learning_to_learn
#### Date: Deadline: Jun 30, 23:59
#### Points: 4 points
#### Examples: learning_to_learn_examples
#### Tests: learning_to_learn_tests

Implement a simple variant of learning-to-learn architecture. Utilizing
the [Omniglot dataset](https://ufal.mff.cuni.cz/~straka/courses/npfl114/2122/demos/omniglot_demo.html)
loadable using the [omniglot_dataset.py](https://github.com/ufal/npfl114/tree/master/labs/14/omniglot_dataset.py)
module, the goal is to learn to classify a
[sequence of images using a custom hierarchy](https://ufal.mff.cuni.cz/~straka/courses/npfl114/2122/demos/learning_to_learn_demo.html)
by employing external memory.

The inputs image sequences consists of `args.classes` random chosen Omniglot
classes, each class being assigned a randomly chosen label. For every chosen
class, `args.images_per_class` images are randomly selected. Apart from the
images, the input contain the random labels one step after the corresponding
images (with the first label being -1). The gold outputs are also the labels,
but without the one-step offset.

The input images should be passed through a CNN feature extraction module
and then processed using memory augmented LSTM controller; the external memory
contains enough memory cells, each with `args.cell_size` units. In each step,
the controller emits:
- `args.read_heads` read keys, each used to perform a read from memory as
  a weighted combination of cells according to the softmax of cosine
  similarities of the read key and the memory cells;
- a write value, which is prepended to the memory (dropping the last cell).

#### Examples Start: learning_to_learn_examples
_Note that your results may be slightly different, depending on your CPU type and whether you use a GPU._
- `python3 learning_to_learn.py --epochs=20 --classes=2`
```
Epoch  1/20 loss: 0.6107 - acc: 0.6307 - acc1: 0.6829 - acc2: 0.6447 - acc5: 0.6230 - acc10: 0.6136 - val_loss: 0.5053 - val_acc: 0.7350 - val_acc1: 0.5465 - val_acc2: 0.6985 - val_acc5: 0.7460 - val_acc10: 0.7995
Epoch  2/20 loss: 0.2924 - acc: 0.8600 - acc1: 0.6087 - acc2: 0.8235 - acc5: 0.8940 - acc10: 0.9068 - val_loss: 0.2914 - val_acc: 0.8624 - val_acc1: 0.6440 - val_acc2: 0.8235 - val_acc5: 0.9000 - val_acc10: 0.9105
Epoch  3/20 loss: 0.1927 - acc: 0.9100 - acc1: 0.6423 - acc2: 0.8874 - acc5: 0.9475 - acc10: 0.9516 - val_loss: 0.2666 - val_acc: 0.8838 - val_acc1: 0.6420 - val_acc2: 0.8535 - val_acc5: 0.9210 - val_acc10: 0.9315
Epoch  4/20 loss: 0.1559 - acc: 0.9263 - acc1: 0.6564 - acc2: 0.9074 - acc5: 0.9633 - acc10: 0.9687 - val_loss: 0.2087 - val_acc: 0.9049 - val_acc1: 0.6270 - val_acc2: 0.8730 - val_acc5: 0.9390 - val_acc10: 0.9545
Epoch  5/20 loss: 0.1349 - acc: 0.9356 - acc1: 0.6691 - acc2: 0.9237 - acc5: 0.9683 - acc10: 0.9761 - val_loss: 0.2170 - val_acc: 0.9061 - val_acc1: 0.6530 - val_acc2: 0.8920 - val_acc5: 0.9355 - val_acc10: 0.9575
Epoch 10/20 loss: 0.0964 - acc: 0.9518 - acc1: 0.6959 - acc2: 0.9554 - acc5: 0.9836 - acc10: 0.9862 - val_loss: 0.1561 - val_acc: 0.9305 - val_acc1: 0.6665 - val_acc2: 0.9220 - val_acc5: 0.9690 - val_acc10: 0.9755
Epoch 15/20 loss: 0.0837 - acc: 0.9565 - acc1: 0.7030 - acc2: 0.9637 - acc5: 0.9864 - acc10: 0.9888 - val_loss: 0.1485 - val_acc: 0.9334 - val_acc1: 0.6695 - val_acc2: 0.9320 - val_acc5: 0.9700 - val_acc10: 0.9745
Epoch 20/20 loss: 0.0745 - acc: 0.9609 - acc1: 0.7139 - acc2: 0.9711 - acc5: 0.9902 - acc10: 0.9924 - val_loss: 0.1971 - val_acc: 0.9196 - val_acc1: 0.6915 - val_acc2: 0.8970 - val_acc5: 0.9455 - val_acc10: 0.9660
```
- `python3 learning_to_learn.py --epochs=20 --read_heads=2 --classes=5`
```
Epoch  1/20 loss: 1.5197 - acc: 0.2864 - acc1: 0.3585 - acc2: 0.2902 - acc5: 0.2742 - acc10: 0.2761 - val_loss: 1.2766 - val_acc: 0.4328 - val_acc1: 0.3390 - val_acc2: 0.3828 - val_acc5: 0.4374 - val_acc10: 0.5014
Epoch  2/20 loss: 0.8115 - acc: 0.6533 - acc1: 0.2820 - acc2: 0.5126 - acc5: 0.7086 - acc10: 0.7579 - val_loss: 0.8253 - val_acc: 0.6546 - val_acc1: 0.2262 - val_acc2: 0.5100 - val_acc5: 0.7132 - val_acc10: 0.7656
Epoch  3/20 loss: 0.5530 - acc: 0.7705 - acc1: 0.2573 - acc2: 0.6250 - acc5: 0.8561 - acc10: 0.8755 - val_loss: 0.7181 - val_acc: 0.7108 - val_acc1: 0.2364 - val_acc2: 0.5494 - val_acc5: 0.7848 - val_acc10: 0.8302
Epoch  5/20 loss: 0.4474 - acc: 0.8171 - acc1: 0.2783 - acc2: 0.7003 - acc5: 0.9011 - acc10: 0.9188 - val_loss: 0.5924 - val_acc: 0.7648 - val_acc1: 0.2682 - val_acc2: 0.6552 - val_acc5: 0.8434 - val_acc10: 0.8568
Epoch 10/20 loss: 0.3356 - acc: 0.8611 - acc1: 0.3086 - acc2: 0.7996 - acc5: 0.9382 - acc10: 0.9466 - val_loss: 0.6684 - val_acc: 0.7719 - val_acc1: 0.3100 - val_acc2: 0.6982 - val_acc5: 0.8192 - val_acc10: 0.8752
Epoch 20/20 loss: 0.2499 - acc: 0.8953 - acc1: 0.3398 - acc2: 0.8851 - acc5: 0.9635 - acc10: 0.9741 - val_loss: 0.5017 - val_acc: 0.8230 - val_acc1: 0.3202 - val_acc2: 0.7908 - val_acc5: 0.8802 - val_acc10: 0.9178
```
#### Examples End:
#### Tests Start: learning_to_learn_tests
_Note that your results may be slightly different, depending on your CPU type and whether you use a GPU._
- `python3 learning_to_learn.py --train_episodes=160 --test_episodes=160 --epochs=3 --classes=2`
```
Epoch 1/3 loss: 0.7764 - acc: 0.5078 - acc1: 0.5375 - acc2: 0.5063 - acc5: 0.5031 - acc10: 0.5000 - val_loss: 0.6923 - val_acc: 0.5175 - val_acc1: 0.7531 - val_acc2: 0.5688 - val_acc5: 0.4500 - val_acc10: 0.4969
Epoch 2/3 loss: 0.6992 - acc: 0.5034 - acc1: 0.5250 - acc2: 0.5031 - acc5: 0.4906 - acc10: 0.5063 - val_loss: 0.6914 - val_acc: 0.5397 - val_acc1: 0.7469 - val_acc2: 0.5844 - val_acc5: 0.5031 - val_acc10: 0.4875
Epoch 3/3 loss: 0.6969 - acc: 0.4975 - acc1: 0.5594 - acc2: 0.5063 - acc5: 0.4844 - acc10: 0.5094 - val_loss: 0.6907 - val_acc: 0.5272 - val_acc1: 0.6781 - val_acc2: 0.5312 - val_acc5: 0.5219 - val_acc10: 0.5000
```
- `python3 learning_to_learn.py --train_episodes=160 --test_episodes=160 --epochs=3 --read_heads=2 --classes=5`
```
Epoch 1/3 loss: 1.6505 - acc: 0.2004 - acc1: 0.1937 - acc2: 0.2025 - acc5: 0.2050 - acc10: 0.2087 - val_loss: 1.6086 - val_acc: 0.2075 - val_acc1: 0.2837 - val_acc2: 0.2325 - val_acc5: 0.1900 - val_acc10: 0.1900
Epoch 2/3 loss: 1.6146 - acc: 0.2042 - acc1: 0.2237 - acc2: 0.1912 - acc5: 0.1950 - acc10: 0.2138 - val_loss: 1.6075 - val_acc: 0.2156 - val_acc1: 0.3050 - val_acc2: 0.2325 - val_acc5: 0.1912 - val_acc10: 0.2100
Epoch 3/3 loss: 1.6114 - acc: 0.2031 - acc1: 0.2275 - acc2: 0.2138 - acc5: 0.1838 - acc10: 0.1912 - val_loss: 1.6061 - val_acc: 0.2261 - val_acc1: 0.3363 - val_acc2: 0.2387 - val_acc5: 0.2163 - val_acc10: 0.2013
```
#### Tests End:
