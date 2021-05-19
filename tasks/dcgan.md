### Assignment: dcgan
#### Date: Deadline: Jun 30, 23:59
#### Points: 1 points
#### Examples: dcgan_examples

This task is a continuation of the `gan` assignment, which you will modify to
implement the Deep Convolutional GAN (DCGAN).

Start with the
[dcgan.py](https://github.com/ufal/npfl114/tree/master/labs/12/dcgan.py)
template and implement a DCGAN. Note that most of the TODO notes are from
the `gan` assignment.

After submitting the assignment to ReCodEx, you can experiment with the three
available datasets (`mnist`, `mnist-fashion`, and `mnist-cifarcars`). However,
note that you will need a lot of computational power (preferably a GPU) to
generate the images; the example outputs below were also generated on a GPU,
which means the results are nondeterministic.

#### Examples Start: dcgan_examples
_Note that your results may be slightly different, depending on your CPU type and whether you use GPU._
- `python3 dcgan.py --dataset=mnist --z_dim=2 --epochs=3`
```
Epoch 1/3 discriminator_loss: 0.2638 - generator_loss: 3.3597 - loss: 0.9523 - discriminator_accuracy: 0.9061
Epoch 2/3 discriminator_loss: 0.0299 - generator_loss: 5.7561 - loss: 1.7968 - discriminator_accuracy: 0.9972
Epoch 3/3 discriminator_loss: 0.0197 - generator_loss: 5.9106 - loss: 1.8184 - discriminator_accuracy: 0.9981
```
- `python3 dcgan.py --dataset=mnist --z_dim=100 --epochs=3`
```
Epoch 1/3 discriminator_loss: 0.2744 - generator_loss: 3.3752 - loss: 0.9341 - discriminator_accuracy: 0.8809
Epoch 2/3 discriminator_loss: 0.0297 - generator_loss: 5.6908 - loss: 1.7981 - discriminator_accuracy: 0.9954
Epoch 3/3 discriminator_loss: 0.0257 - generator_loss: 6.2856 - loss: 2.1166 - discriminator_accuracy: 0.9974
```
- `python3 dcgan.py --dataset=mnist-fashion --z_dim=2 --epochs=3`
```
Epoch 1/3 discriminator_loss: 0.3830 - generator_loss: 2.5970 - loss: 0.8996 - discriminator_accuracy: 0.9198
Epoch 2/3 discriminator_loss: 0.2759 - generator_loss: 3.3412 - loss: 1.1519 - discriminator_accuracy: 0.9545
Epoch 3/3 discriminator_loss: 0.2125 - generator_loss: 3.9514 - loss: 1.3584 - discriminator_accuracy: 0.9681
```
- `python3 dcgan.py --dataset=mnist-fashion --z_dim=100 --epochs=3`
```
Epoch 1/3 discriminator_loss: 0.4766 - generator_loss: 2.4001 - loss: 0.8588 - discriminator_accuracy: 0.8763
Epoch 2/3 discriminator_loss: 0.4254 - generator_loss: 2.8352 - loss: 1.0735 - discriminator_accuracy: 0.9250
Epoch 3/3 discriminator_loss: 0.3939 - generator_loss: 3.0114 - loss: 1.1252 - discriminator_accuracy: 0.9285
```
- `python3 dcgan.py --dataset=mnist-cifarcars --z_dim=2 --epochs=3`
```
Epoch 1/3 discriminator_loss: 0.8294 - generator_loss: 1.4831 - loss: 0.7460 - discriminator_accuracy: 0.7689
Epoch 2/3 discriminator_loss: 0.4352 - generator_loss: 2.4002 - loss: 0.9303 - discriminator_accuracy: 0.9297
Epoch 3/3 discriminator_loss: 0.3052 - generator_loss: 3.0020 - loss: 1.0943 - discriminator_accuracy: 0.9627
```
- `python3 dcgan.py --dataset=mnist-cifarcars --z_dim=100 --epochs=3`
```
Epoch 1/3 discriminator_loss: 1.1401 - generator_loss: 1.0359 - loss: 0.7335 - discriminator_accuracy: 0.6756
Epoch 2/3 discriminator_loss: 0.8321 - generator_loss: 1.5365 - loss: 0.7724 - discriminator_accuracy: 0.7945
Epoch 3/3 discriminator_loss: 0.5566 - generator_loss: 2.2292 - loss: 0.9219 - discriminator_accuracy: 0.8965
```
#### Examples End:
