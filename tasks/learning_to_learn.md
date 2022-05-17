### Assignment: learning_to_learn
#### Date: Deadline: Jun 30, 23:59
#### Points: 4 points
#### Examples: learning_to_learn_examples
#### Tests: learning_to_learn_tests

Implement a simple variant of learning-to-learn architecture using the
[learning_to_learn.py](https://github.com/ufal/npfl114/tree/master/labs/14/learning_to_learn.py)
template. Utilizing the [Omniglot dataset](https://ufal.mff.cuni.cz/~straka/courses/npfl114/2122/demos/omniglot_demo.html)
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
- `python3 learning_to_learn.py --epochs=50 --classes=2`
```
Epoch  1/50 loss: 0.5964 - acc: 0.6422 - acc1: 0.6768 - acc2: 0.6529 - acc5: 0.6360 - acc10: 0.6299 - val_loss: 0.4335 - val_acc: 0.7825 - val_acc1: 0.5195 - val_acc2: 0.7460 - val_acc5: 0.8010 - val_acc10: 0.8725
Epoch  2/50 loss: 0.2660 - acc: 0.8759 - acc1: 0.6137 - acc2: 0.8349 - acc5: 0.9097 - acc10: 0.9272 - val_loss: 0.2693 - val_acc: 0.8744 - val_acc1: 0.6390 - val_acc2: 0.8225 - val_acc5: 0.9135 - val_acc10: 0.9275
Epoch  3/50 loss: 0.1792 - acc: 0.9164 - acc1: 0.6364 - acc2: 0.8870 - acc5: 0.9550 - acc10: 0.9609 - val_loss: 0.2980 - val_acc: 0.8770 - val_acc1: 0.6250 - val_acc2: 0.8460 - val_acc5: 0.9130 - val_acc10: 0.9260
Epoch  4/50 loss: 0.1504 - acc: 0.9296 - acc1: 0.6544 - acc2: 0.9133 - acc5: 0.9643 - acc10: 0.9725 - val_loss: 0.2037 - val_acc: 0.9083 - val_acc1: 0.6390 - val_acc2: 0.8860 - val_acc5: 0.9420 - val_acc10: 0.9555
Epoch  5/50 loss: 0.1326 - acc: 0.9369 - acc1: 0.6741 - acc2: 0.9261 - acc5: 0.9686 - acc10: 0.9772 - val_loss: 0.1829 - val_acc: 0.9168 - val_acc1: 0.6485 - val_acc2: 0.9025 - val_acc5: 0.9560 - val_acc10: 0.9600
Epoch 10/50 loss: 0.0952 - acc: 0.9525 - acc1: 0.6985 - acc2: 0.9542 - acc5: 0.9825 - acc10: 0.9880 - val_loss: 0.1709 - val_acc: 0.9240 - val_acc1: 0.6280 - val_acc2: 0.9140 - val_acc5: 0.9655 - val_acc10: 0.9685
Epoch 20/50 loss: 0.0729 - acc: 0.9613 - acc1: 0.7106 - acc2: 0.9732 - acc5: 0.9916 - acc10: 0.9937 - val_loss: 0.1401 - val_acc: 0.9383 - val_acc1: 0.6845 - val_acc2: 0.9310 - val_acc5: 0.9690 - val_acc10: 0.9805
Epoch 50/50 loss: 0.0579 - acc: 0.9668 - acc1: 0.7243 - acc2: 0.9833 - acc5: 0.9948 - acc10: 0.9961 - val_loss: 0.1271 - val_acc: 0.9444 - val_acc1: 0.7110 - val_acc2: 0.9385 - val_acc5: 0.9760 - val_acc10: 0.9835
```
- `python3 learning_to_learn.py --epochs=50 --read_heads=2 --classes=5`
```
Epoch  1/50 loss: 1.5479 - acc: 0.2698 - acc1: 0.3502 - acc2: 0.2777 - acc5: 0.2588 - acc10: 0.2571 - val_loss: 1.4092 - val_acc: 0.3719 - val_acc1: 0.3176 - val_acc2: 0.3430 - val_acc5: 0.3568 - val_acc10: 0.4202
Epoch  2/50 loss: 0.8753 - acc: 0.6209 - acc1: 0.2889 - acc2: 0.4895 - acc5: 0.6703 - acc10: 0.7216 - val_loss: 0.7641 - val_acc: 0.6890 - val_acc1: 0.2538 - val_acc2: 0.5340 - val_acc5: 0.7508 - val_acc10: 0.8050
Epoch  3/50 loss: 0.5346 - acc: 0.7813 - acc1: 0.2553 - acc2: 0.6352 - acc5: 0.8657 - acc10: 0.8919 - val_loss: 0.6430 - val_acc: 0.7511 - val_acc1: 0.2608 - val_acc2: 0.6134 - val_acc5: 0.8286 - val_acc10: 0.8614
Epoch  4/50 loss: 0.4314 - acc: 0.8231 - acc1: 0.2716 - acc2: 0.6970 - acc5: 0.9090 - acc10: 0.9250 - val_loss: 0.5841 - val_acc: 0.7696 - val_acc1: 0.2796 - val_acc2: 0.6414 - val_acc5: 0.8390 - val_acc10: 0.8760
Epoch  5/50 loss: 0.3852 - acc: 0.8410 - acc1: 0.2851 - acc2: 0.7280 - acc5: 0.9260 - acc10: 0.9400 - val_loss: 0.7275 - val_acc: 0.7390 - val_acc1: 0.2836 - val_acc2: 0.6138 - val_acc5: 0.8024 - val_acc10: 0.8456
Epoch 10/50 loss: 0.2885 - acc: 0.8799 - acc1: 0.3195 - acc2: 0.8274 - acc5: 0.9569 - acc10: 0.9656 - val_loss: 0.8520 - val_acc: 0.7335 - val_acc1: 0.2994 - val_acc2: 0.6314 - val_acc5: 0.7852 - val_acc10: 0.8416
Epoch 20/50 loss: 0.2252 - acc: 0.9049 - acc1: 0.3511 - acc2: 0.9009 - acc5: 0.9750 - acc10: 0.9805 - val_loss: 0.5483 - val_acc: 0.8216 - val_acc1: 0.3182 - val_acc2: 0.7828 - val_acc5: 0.8828 - val_acc10: 0.9152
Epoch 50/50 loss: 0.1720 - acc: 0.9233 - acc1: 0.3859 - acc2: 0.9518 - acc5: 0.9870 - acc10: 0.9895 - val_loss: 0.5175 - val_acc: 0.8478 - val_acc1: 0.3636 - val_acc2: 0.8288 - val_acc5: 0.9006 - val_acc10: 0.9324
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
