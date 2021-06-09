### Assignment: learning_to_learn
#### Date: Deadline: Jun 30, 23:59
#### Points: 4 points
#### Tests: learning_to_learn_tests
#### Examples: learning_to_learn_examples

Implement a simple variant of learning-to-learn architecture. Utilizing
the [Omniglot dataset](https://ufal.mff.cuni.cz/~straka/courses/npfl114/2021/demos/omniglot_demo.html)
loadable using the [omniglot_dataset.py](https://github.com/ufal/npfl114/tree/master/labs/14/omniglot_dataset.py)
module, the goal is to learn to classify a
[sequence of images using a custom hierarchy](https://ufal.mff.cuni.cz/~straka/courses/npfl114/2021/demos/learning_to_learn_demo.html)
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

#### Tests Start: learning_to_learn_tests
_These tests are identical to the ones in ReCodEx, apart from a different random seed.
Note that your results may be slightly different, depending on your CPU type and whether you use GPU._
- `python3 learning_to_learn.py --recodex --train_episodes=160 --test_episodes=160 --epochs=3 --classes=2`
```
Epoch 1/3 loss: 0.8135 - acc: 0.5100 - acc1: 0.5254 - acc2: 0.5250 - acc5: 0.5102 - acc10: 0.5086 - val_loss: 0.6928 - val_acc: 0.5000 - val_acc1: 0.5000 - val_acc2: 0.5000 - val_acc5: 0.5000 - val_acc10: 0.5000
Epoch 2/3 loss: 0.7014 - acc: 0.4985 - acc1: 0.4974 - acc2: 0.4868 - acc5: 0.4918 - acc10: 0.5170 - val_loss: 0.6914 - val_acc: 0.5519 - val_acc1: 0.7750 - val_acc2: 0.6313 - val_acc5: 0.5094 - val_acc10: 0.4719
Epoch 3/3 loss: 0.6932 - acc: 0.5046 - acc1: 0.5233 - acc2: 0.4772 - acc5: 0.5386 - acc10: 0.5403 - val_loss: 0.6902 - val_acc: 0.5419 - val_acc1: 0.7500 - val_acc2: 0.6125 - val_acc5: 0.4844 - val_acc10: 0.4781
```
- `python3 learning_to_learn.py --recodex --train_episodes=160 --test_episodes=160 --epochs=3 --classes=5`
```
Epoch 1/3 loss: 1.6601 - acc: 0.1991 - acc1: 0.2227 - acc2: 0.1895 - acc5: 0.1909 - acc10: 0.2050 - val_loss: 1.6095 - val_acc: 0.2075 - val_acc1: 0.2150 - val_acc2: 0.2313 - val_acc5: 0.2025 - val_acc10: 0.1900
Epoch 2/3 loss: 1.6168 - acc: 0.2086 - acc1: 0.2090 - acc2: 0.2406 - acc5: 0.2214 - acc10: 0.2048 - val_loss: 1.6079 - val_acc: 0.2025 - val_acc1: 0.2512 - val_acc2: 0.2113 - val_acc5: 0.1925 - val_acc10: 0.1887
Epoch 3/3 loss: 1.6129 - acc: 0.2108 - acc1: 0.2369 - acc2: 0.2266 - acc5: 0.1965 - acc10: 0.2131 - val_loss: 1.6066 - val_acc: 0.2191 - val_acc1: 0.3288 - val_acc2: 0.2237 - val_acc5: 0.2025 - val_acc10: 0.2000
```
#### Tests End:

#### Examples Start: learning_to_learn_examples
_Note that your results may be slightly different, depending on your CPU type and whether you use GPU._
- `python3 learning_to_learn.py --classes=2 --epochs=15`
```
Epoch 1/15 loss: 0.6767 - acc: 0.5680 - acc1: 0.6775 - acc2: 0.5956 - acc5: 0.5540 - acc10: 0.5338 - val_loss: 0.5088 - val_acc: 0.7240 - val_acc1: 0.5565 - val_acc2: 0.6775 - val_acc5: 0.7350 - val_acc10: 0.7915
Epoch 2/15 loss: 0.3449 - acc: 0.8293 - acc1: 0.6079 - acc2: 0.7752 - acc5: 0.8608 - acc10: 0.8790 - val_loss: 0.2999 - val_acc: 0.8583 - val_acc1: 0.6225 - val_acc2: 0.8440 - val_acc5: 0.8990 - val_acc10: 0.9040
Epoch 3/15 loss: 0.2161 - acc: 0.8982 - acc1: 0.6318 - acc2: 0.8698 - acc5: 0.9345 - acc10: 0.9407 - val_loss: 0.2402 - val_acc: 0.8911 - val_acc1: 0.6370 - val_acc2: 0.8700 - val_acc5: 0.9230 - val_acc10: 0.9430
Epoch 4/15 loss: 0.1722 - acc: 0.9196 - acc1: 0.6522 - acc2: 0.9000 - acc5: 0.9547 - acc10: 0.9618 - val_loss: 0.2731 - val_acc: 0.8781 - val_acc1: 0.5670 - val_acc2: 0.8445 - val_acc5: 0.9180 - val_acc10: 0.9485
Epoch 5/15 loss: 0.1368 - acc: 0.9351 - acc1: 0.6626 - acc2: 0.9252 - acc5: 0.9681 - acc10: 0.9747 - val_loss: 0.2428 - val_acc: 0.8996 - val_acc1: 0.6405 - val_acc2: 0.8850 - val_acc5: 0.9320 - val_acc10: 0.9555
Epoch 10/15 loss: 0.0998 - acc: 0.9508 - acc1: 0.6963 - acc2: 0.9541 - acc5: 0.9841 - acc10: 0.9860 - val_loss: 0.1809 - val_acc: 0.9183 - val_acc1: 0.6525 - val_acc2: 0.8845 - val_acc5: 0.9560 - val_acc10: 0.9685
Epoch 15/15 loss: 0.0838 - acc: 0.9574 - acc1: 0.7111 - acc2: 0.9627 - acc5: 0.9862 - acc10: 0.9890 - val_loss: 0.1396 - val_acc: 0.9363 - val_acc1: 0.6730 - val_acc2: 0.9420 - val_acc5: 0.9685 - val_acc10: 0.9745
```
- `python3 learning_to_learn.py --classes=5 --epochs=20`
```
Epoch 1/20 loss: 1.5986 - acc: 0.2322 - acc1: 0.3207 - acc2: 0.2482 - acc5: 0.2216 - acc10: 0.2105 - val_loss: 1.3823 - val_acc: 0.3742 - val_acc1: 0.3314 - val_acc2: 0.3510 - val_acc5: 0.3590 - val_acc10: 0.4216
Epoch 2/20 loss: 1.0789 - acc: 0.5180 - acc1: 0.3300 - acc2: 0.4259 - acc5: 0.5362 - acc10: 0.5986 - val_loss: 0.8358 - val_acc: 0.6513 - val_acc1: 0.2540 - val_acc2: 0.5272 - val_acc5: 0.7080 - val_acc10: 0.7510
Epoch 3/20 loss: 0.6539 - acc: 0.7278 - acc1: 0.2490 - acc2: 0.5778 - acc5: 0.8089 - acc10: 0.8333 - val_loss: 0.7388 - val_acc: 0.7011 - val_acc1: 0.2322 - val_acc2: 0.5658 - val_acc5: 0.7780 - val_acc10: 0.7960
Epoch 4/20 loss: 0.5285 - acc: 0.7808 - acc1: 0.2587 - acc2: 0.6480 - acc5: 0.8639 - acc10: 0.8809 - val_loss: 0.7356 - val_acc: 0.7230 - val_acc1: 0.2710 - val_acc2: 0.5962 - val_acc5: 0.7856 - val_acc10: 0.8222
Epoch 5/20 loss: 0.4426 - acc: 0.8190 - acc1: 0.2771 - acc2: 0.7031 - acc5: 0.9014 - acc10: 0.9179 - val_loss: 0.5951 - val_acc: 0.7729 - val_acc1: 0.2810 - val_acc2: 0.6618 - val_acc5: 0.8446 - val_acc10: 0.8666
Epoch 10/20 loss: 0.3195 - acc: 0.8676 - acc1: 0.3084 - acc2: 0.8063 - acc5: 0.9467 - acc10: 0.9547 - val_loss: 0.6660 - val_acc: 0.7694 - val_acc1: 0.3270 - val_acc2: 0.6970 - val_acc5: 0.8218 - val_acc10: 0.8634
Epoch 15/20 loss: 0.2625 - acc: 0.8902 - acc1: 0.3336 - acc2: 0.8661 - acc5: 0.9622 - acc10: 0.9714 - val_loss: 0.6239 - val_acc: 0.7967 - val_acc1: 0.3088 - val_acc2: 0.7498 - val_acc5: 0.8534 - val_acc10: 0.8832
Epoch 20/20 loss: 0.2446 - acc: 0.8970 - acc1: 0.3427 - acc2: 0.8872 - acc5: 0.9663 - acc10: 0.9733 - val_loss: 0.5438 - val_acc: 0.8203 - val_acc1: 0.3174 - val_acc2: 0.7984 - val_acc5: 0.8766 - val_acc10: 0.9036
```
#### Examples End:
