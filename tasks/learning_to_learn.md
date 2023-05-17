### Assignment: learning_to_learn
#### Date: Deadline: Jun 30, 23:59
#### Points: 4 points
#### Tests: learning_to_learn_tests
#### Examples: learning_to_learn_examples

Implement a simple variant of learning-to-learn architecture using the
[learning_to_learn.py](https://github.com/ufal/npfl114/tree/master/labs/14/learning_to_learn.py)
template. Utilizing the [Omniglot dataset](https://ufal.mff.cuni.cz/~straka/courses/npfl114/2223/demos/omniglot_demo.html)
loadable using the [omniglot_dataset.py](https://github.com/ufal/npfl114/tree/master/labs/14/omniglot_dataset.py)
module, the goal is to learn to classify a
[sequence of images using a custom hierarchy](https://ufal.mff.cuni.cz/~straka/courses/npfl114/2223/demos/learning_to_learn_demo.html)
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
_Note that your results may be slightly different, depending on your CPU type and whether you use a GPU._
1. `python3 learning_to_learn.py --train_episodes=160 --test_episodes=160 --epochs=3 --classes=2`
```
Epoch 1/3 loss: 0.7535 - acc: 0.4984 - acc1: 0.5250 - acc2: 0.4875 - acc5: 0.4938 - acc10: 0.5000 - val_loss: 0.6918 - val_acc: 0.5525 - val_acc1: 0.7375 - val_acc2: 0.6125 - val_acc5: 0.5531 - val_acc10: 0.4969
Epoch 2/3 loss: 0.6968 - acc: 0.4956 - acc1: 0.5531 - acc2: 0.4969 - acc5: 0.5031 - acc10: 0.4719 - val_loss: 0.6907 - val_acc: 0.5447 - val_acc1: 0.6969 - val_acc2: 0.6187 - val_acc5: 0.5344 - val_acc10: 0.4906
Epoch 3/3 loss: 0.6937 - acc: 0.5138 - acc1: 0.5781 - acc2: 0.5094 - acc5: 0.5125 - acc10: 0.4812 - val_loss: 0.6895 - val_acc: 0.5547 - val_acc1: 0.7688 - val_acc2: 0.5938 - val_acc5: 0.5063 - val_acc10: 0.4875
```
2. `python3 learning_to_learn.py --train_episodes=160 --test_episodes=160 --epochs=3 --read_heads=2 --classes=5`
```
Epoch 1/3 loss: 1.6529 - acc: 0.2004 - acc1: 0.2050 - acc2: 0.1838 - acc5: 0.2100 - acc10: 0.2075 - val_loss: 1.6091 - val_acc: 0.2136 - val_acc1: 0.2812 - val_acc2: 0.2100 - val_acc5: 0.2013 - val_acc10: 0.1925
Epoch 2/3 loss: 1.6139 - acc: 0.1996 - acc1: 0.2113 - acc2: 0.1675 - acc5: 0.2025 - acc10: 0.1925 - val_loss: 1.6078 - val_acc: 0.1984 - val_acc1: 0.2125 - val_acc2: 0.2075 - val_acc5: 0.2075 - val_acc10: 0.1850
Epoch 3/3 loss: 1.6102 - acc: 0.2066 - acc1: 0.2200 - acc2: 0.2150 - acc5: 0.2138 - acc10: 0.2013 - val_loss: 1.6068 - val_acc: 0.2237 - val_acc1: 0.3988 - val_acc2: 0.2188 - val_acc5: 0.2100 - val_acc10: 0.1688
```
#### Tests End:
#### Examples Start: learning_to_learn_examples
_Note that your results may be slightly different, depending on your CPU type and whether you use a GPU._
- `python3 learning_to_learn.py --epochs=50 --classes=2`
```
Epoch 1/50 loss: 0.6592 - acc: 0.5888 - acc1: 0.7211 - acc2: 0.6151 - acc5: 0.5750 - acc10: 0.5482 - val_loss: 0.6233 - val_acc: 0.6270 - val_acc1: 0.7290 - val_acc2: 0.6310 - val_acc5: 0.5960 - val_acc10: 0.6510
Epoch 2/50 loss: 0.4043 - acc: 0.7890 - acc1: 0.6024 - acc2: 0.7499 - acc5: 0.8166 - acc10: 0.8218 - val_loss: 0.3930 - val_acc: 0.8067 - val_acc1: 0.6135 - val_acc2: 0.7830 - val_acc5: 0.8485 - val_acc10: 0.8485
Epoch 3/50 loss: 0.2931 - acc: 0.8566 - acc1: 0.6158 - acc2: 0.8373 - acc5: 0.8928 - acc10: 0.8890 - val_loss: 0.3038 - val_acc: 0.8551 - val_acc1: 0.6165 - val_acc2: 0.8275 - val_acc5: 0.8990 - val_acc10: 0.9060
Epoch 4/50 loss: 0.2133 - acc: 0.8988 - acc1: 0.6371 - acc2: 0.8763 - acc5: 0.9320 - acc10: 0.9391 - val_loss: 0.2513 - val_acc: 0.8849 - val_acc1: 0.6365 - val_acc2: 0.8440 - val_acc5: 0.9155 - val_acc10: 0.9360
Epoch 5/50 loss: 0.1714 - acc: 0.9194 - acc1: 0.6637 - acc2: 0.9099 - acc5: 0.9480 - acc10: 0.9561 - val_loss: 0.2459 - val_acc: 0.8897 - val_acc1: 0.6235 - val_acc2: 0.8750 - val_acc5: 0.9265 - val_acc10: 0.9400
Epoch 10/50 loss: 0.1125 - acc: 0.9449 - acc1: 0.6888 - acc2: 0.9461 - acc5: 0.9754 - acc10: 0.9801 - val_loss: 0.1714 - val_acc: 0.9201 - val_acc1: 0.6665 - val_acc2: 0.8975 - val_acc5: 0.9535 - val_acc10: 0.9665
Epoch 20/50 loss: 0.0784 - acc: 0.9588 - acc1: 0.7069 - acc2: 0.9671 - acc5: 0.9891 - acc10: 0.9908 - val_loss: 0.1525 - val_acc: 0.9320 - val_acc1: 0.6720 - val_acc2: 0.9255 - val_acc5: 0.9640 - val_acc10: 0.9755
Epoch 50/50 loss: 0.0585 - acc: 0.9659 - acc1: 0.7199 - acc2: 0.9819 - acc5: 0.9950 - acc10: 0.9958 - val_loss: 0.1255 - val_acc: 0.9430 - val_acc1: 0.6900 - val_acc2: 0.9280 - val_acc5: 0.9760 - val_acc10: 0.9860
```
- `python3 learning_to_learn.py --epochs=50 --read_heads=2 --classes=5`
```
Epoch 1/50 loss: 1.5718 - acc: 0.2606 - acc1: 0.3784 - acc2: 0.2764 - acc5: 0.2469 - acc10: 0.2316 - val_loss: 1.3803 - val_acc: 0.3761 - val_acc1: 0.4078 - val_acc2: 0.3412 - val_acc5: 0.3638 - val_acc10: 0.4156
Epoch 2/50 loss: 0.9240 - acc: 0.5968 - acc1: 0.2956 - acc2: 0.4762 - acc5: 0.6400 - acc10: 0.6911 - val_loss: 0.8087 - val_acc: 0.6602 - val_acc1: 0.2484 - val_acc2: 0.5250 - val_acc5: 0.7178 - val_acc10: 0.7606
Epoch 3/50 loss: 0.5984 - acc: 0.7496 - acc1: 0.2499 - acc2: 0.6078 - acc5: 0.8306 - acc10: 0.8504 - val_loss: 0.7452 - val_acc: 0.6980 - val_acc1: 0.2322 - val_acc2: 0.5628 - val_acc5: 0.7754 - val_acc10: 0.8008
Epoch 4/50 loss: 0.5145 - acc: 0.7851 - acc1: 0.2579 - acc2: 0.6571 - acc5: 0.8713 - acc10: 0.8807 - val_loss: 0.7892 - val_acc: 0.7053 - val_acc1: 0.2684 - val_acc2: 0.5882 - val_acc5: 0.7708 - val_acc10: 0.8040
Epoch 5/50 loss: 0.4730 - acc: 0.8016 - acc1: 0.2680 - acc2: 0.6903 - acc5: 0.8865 - acc10: 0.8941 - val_loss: 0.6920 - val_acc: 0.7294 - val_acc1: 0.2722 - val_acc2: 0.6196 - val_acc5: 0.7964 - val_acc10: 0.8158
Epoch 10/50 loss: 0.3415 - acc: 0.8568 - acc1: 0.3011 - acc2: 0.7917 - acc5: 0.9351 - acc10: 0.9435 - val_loss: 0.6062 - val_acc: 0.7794 - val_acc1: 0.2922 - val_acc2: 0.6944 - val_acc5: 0.8428 - val_acc10: 0.8688
Epoch 20/50 loss: 0.2468 - acc: 0.8956 - acc1: 0.3413 - acc2: 0.8869 - acc5: 0.9649 - acc10: 0.9727 - val_loss: 0.5414 - val_acc: 0.8173 - val_acc1: 0.3004 - val_acc2: 0.7856 - val_acc5: 0.8782 - val_acc10: 0.9050
Epoch 50/50 loss: 0.1778 - acc: 0.9210 - acc1: 0.3793 - acc2: 0.9479 - acc5: 0.9850 - acc10: 0.9888 - val_loss: 0.5219 - val_acc: 0.8433 - val_acc1: 0.3728 - val_acc2: 0.8226 - val_acc5: 0.8990 - val_acc10: 0.9346
```
#### Examples End:
