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
Epoch 2/3 loss: 0.7014 - acc: 0.4985 - acc1: 0.4974 - acc2: 0.4868 - acc5: 0.4918 - acc10: 0.5170 - val_loss: 0.6914 - val_acc: 0.5522 - val_acc1: 0.7750 - val_acc2: 0.6344 - val_acc5: 0.5125 - val_acc10: 0.4719
Epoch 3/3 loss: 0.6932 - acc: 0.5045 - acc1: 0.5233 - acc2: 0.4772 - acc5: 0.5386 - acc10: 0.5403 - val_loss: 0.6902 - val_acc: 0.5416 - val_acc1: 0.7500 - val_acc2: 0.6125 - val_acc5: 0.4844 - val_acc10: 0.4781
```
- `python3 learning_to_learn.py --recodex --train_episodes=160 --test_episodes=160 --epochs=3 --classes=5`
```
Epoch 1/3 loss: 1.6601 - acc: 0.1993 - acc1: 0.2227 - acc2: 0.1895 - acc5: 0.1909 - acc10: 0.2063 - val_loss: 1.6094 - val_acc: 0.2077 - val_acc1: 0.2163 - val_acc2: 0.2313 - val_acc5: 0.2013 - val_acc10: 0.1900
Epoch 2/3 loss: 1.6168 - acc: 0.2089 - acc1: 0.2090 - acc2: 0.2406 - acc5: 0.2214 - acc10: 0.2048 - val_loss: 1.6079 - val_acc: 0.2027 - val_acc1: 0.2500 - val_acc2: 0.2125 - val_acc5: 0.1937 - val_acc10: 0.1900
Epoch 3/3 loss: 1.6129 - acc: 0.2111 - acc1: 0.2369 - acc2: 0.2266 - acc5: 0.1976 - acc10: 0.2131 - val_loss: 1.6066 - val_acc: 0.2184 - val_acc1: 0.3237 - val_acc2: 0.2237 - val_acc5: 0.2025 - val_acc10: 0.2000
```
#### Tests End:

#### Examples Start: learning_to_learn_examples
_Note that your results may be slightly different, depending on your CPU type and whether you use GPU._
- `python3 learning_to_learn.py --classes=2 --epochs=20`
```
Epoch 1/20 loss: 0.6769 - acc: 0.5682 - acc1: 0.6769 - acc2: 0.5943 - acc5: 0.5546 - acc10: 0.5331 - val_loss: 0.4930 - val_acc: 0.7337 - val_acc1: 0.5415 - val_acc2: 0.6910 - val_acc5: 0.7525 - val_acc10: 0.8065
Epoch 2/20 loss: 0.3461 - acc: 0.8278 - acc1: 0.6054 - acc2: 0.7646 - acc5: 0.8629 - acc10: 0.8790 - val_loss: 0.2857 - val_acc: 0.8681 - val_acc1: 0.6345 - val_acc2: 0.8355 - val_acc5: 0.9050 - val_acc10: 0.9270
Epoch 3/20 loss: 0.2061 - acc: 0.9045 - acc1: 0.6381 - acc2: 0.8721 - acc5: 0.9407 - acc10: 0.9458 - val_loss: 0.2420 - val_acc: 0.8895 - val_acc1: 0.6160 - val_acc2: 0.8435 - val_acc5: 0.9295 - val_acc10: 0.9505
Epoch 4/20 loss: 0.1619 - acc: 0.9242 - acc1: 0.6459 - acc2: 0.9057 - acc5: 0.9607 - acc10: 0.9680 - val_loss: 0.1938 - val_acc: 0.9122 - val_acc1: 0.6420 - val_acc2: 0.8815 - val_acc5: 0.9585 - val_acc10: 0.9630
Epoch 5/20 loss: 0.1340 - acc: 0.9363 - acc1: 0.6693 - acc2: 0.9237 - acc5: 0.9692 - acc10: 0.9768 - val_loss: 0.2057 - val_acc: 0.9099 - val_acc1: 0.6735 - val_acc2: 0.8870 - val_acc5: 0.9405 - val_acc10: 0.9540
Epoch 10/20 loss: 0.0998 - acc: 0.9510 - acc1: 0.6949 - acc2: 0.9545 - acc5: 0.9833 - acc10: 0.9855 - val_loss: 0.1590 - val_acc: 0.9273 - val_acc1: 0.6585 - val_acc2: 0.9055 - val_acc5: 0.9690 - val_acc10: 0.9735
Epoch 20/20 loss: 0.0739 - acc: 0.9604 - acc1: 0.7074 - acc2: 0.9712 - acc5: 0.9913 - acc10: 0.9937 - val_loss: 0.1510 - val_acc: 0.9356 - val_acc1: 0.6815 - val_acc2: 0.9270 - val_acc5: 0.9665 - val_acc10: 0.9785
```
- `python3 learning_to_learn.py --classes=5 --epochs=20`
```
Epoch 1/20 loss: 1.6013 - acc: 0.2300 - acc1: 0.3162 - acc2: 0.2454 - acc5: 0.2198 - acc10: 0.2094 - val_loss: 1.3712 - val_acc: 0.3809 - val_acc1: 0.3884 - val_acc2: 0.3504 - val_acc5: 0.3692 - val_acc10: 0.4240
Epoch 2/20 loss: 1.1060 - acc: 0.5052 - acc1: 0.3377 - acc2: 0.4164 - acc5: 0.5215 - acc10: 0.5802 - val_loss: 0.8220 - val_acc: 0.6575 - val_acc1: 0.2498 - val_acc2: 0.5318 - val_acc5: 0.7168 - val_acc10: 0.7626
Epoch 3/20 loss: 0.6655 - acc: 0.7209 - acc1: 0.2486 - acc2: 0.5665 - acc5: 0.7999 - acc10: 0.8255 - val_loss: 0.8701 - val_acc: 0.6682 - val_acc1: 0.2568 - val_acc2: 0.5396 - val_acc5: 0.7256 - val_acc10: 0.7730
Epoch 4/20 loss: 0.5154 - acc: 0.7879 - acc1: 0.2612 - acc2: 0.6505 - acc5: 0.8734 - acc10: 0.8924 - val_loss: 0.6253 - val_acc: 0.7506 - val_acc1: 0.2554 - val_acc2: 0.6304 - val_acc5: 0.8302 - val_acc10: 0.8462
Epoch 5/20 loss: 0.4474 - acc: 0.8171 - acc1: 0.2783 - acc2: 0.7003 - acc5: 0.9011 - acc10: 0.9188 - val_loss: 0.5924 - val_acc: 0.7648 - val_acc1: 0.2682 - val_acc2: 0.6552 - val_acc5: 0.8434 - val_acc10: 0.8568
Epoch 10/20 loss: 0.3356 - acc: 0.8611 - acc1: 0.3086 - acc2: 0.7996 - acc5: 0.9382 - acc10: 0.9466 - val_loss: 0.6684 - val_acc: 0.7719 - val_acc1: 0.3100 - val_acc2: 0.6982 - val_acc5: 0.8192 - val_acc10: 0.8752
Epoch 20/20 loss: 0.2499 - acc: 0.8953 - acc1: 0.3398 - acc2: 0.8851 - acc5: 0.9635 - acc10: 0.9741 - val_loss: 0.5017 - val_acc: 0.8230 - val_acc1: 0.3202 - val_acc2: 0.7908 - val_acc5: 0.8802 - val_acc10: 0.9178
```
#### Examples End:
