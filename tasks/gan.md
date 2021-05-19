### Assignment: gan
#### Date: Deadline: Jun 30, 23:59
#### Points: 2 points
#### Examples: gan_examples

In this assignment you will implement a simple Generative Adversarion Network
for three datasets in the MNIST format. Your goal is to modify the
[gan.py](https://github.com/ufal/npfl114/tree/master/labs/12/gan.py)
template and implement a GAN.

After submitting the assignment to ReCodEx, you can experiment with the three
available datasets (`mnist`, `mnist-fashion`, and `mnist-cifarcars`) and
maybe try different latent variable dimensionality. The generated images are
available in TensorBoard logs.

You can also continue with `dcgan` assignment.

#### Examples Start: gan_examples
_Note that your results may be slightly different, depending on your CPU type and whether you use GPU._
- `python3 gan.py --dataset=mnist --z_dim=2 --epochs=5`
```
Epoch 1/5 discriminator_loss: 0.0811 - generator_loss: 5.2954 - loss: 1.7356 - discriminator_accuracy: 0.9826
Epoch 2/5 discriminator_loss: 0.0776 - generator_loss: 3.8221 - loss: 1.3290 - discriminator_accuracy: 0.9926
Epoch 3/5 discriminator_loss: 0.0686 - generator_loss: 4.3589 - loss: 1.3821 - discriminator_accuracy: 0.9920
Epoch 4/5 discriminator_loss: 0.0694 - generator_loss: 4.4692 - loss: 1.4952 - discriminator_accuracy: 0.9910
Epoch 5/5 discriminator_loss: 0.0668 - generator_loss: 4.5452 - loss: 1.5248 - discriminator_accuracy: 0.9919
```
- `python3 gan.py --dataset=mnist --z_dim=100 --epochs=5`
```
Epoch 1/5 discriminator_loss: 0.0526 - generator_loss: 5.6836 - loss: 1.5494 - discriminator_accuracy: 0.9826
Epoch 2/5 discriminator_loss: 0.0333 - generator_loss: 5.9819 - loss: 1.9048 - discriminator_accuracy: 0.9978
Epoch 3/5 discriminator_loss: 0.0660 - generator_loss: 5.0259 - loss: 1.7150 - discriminator_accuracy: 0.9934
Epoch 4/5 discriminator_loss: 0.1227 - generator_loss: 4.9251 - loss: 1.8218 - discriminator_accuracy: 0.9871
Epoch 5/5 discriminator_loss: 0.2496 - generator_loss: 4.0308 - loss: 1.4528 - discriminator_accuracy: 0.9609
```
- `python3 gan.py --dataset=mnist-fashion --z_dim=2 --epochs=5`
```
Epoch 1/5 discriminator_loss: 0.1560 - generator_loss: 12.4313 - loss: 1.6760 - discriminator_accuracy: 0.9788
Epoch 2/5 discriminator_loss: 0.1748 - generator_loss: 21.1818 - loss: 10.1500 - discriminator_accuracy: 0.9644
Epoch 3/5 discriminator_loss: 0.0691 - generator_loss: 11.8005 - loss: 5.7323 - discriminator_accuracy: 0.9919
Epoch 4/5 discriminator_loss: 0.0429 - generator_loss: 15.0839 - loss: 5.9234 - discriminator_accuracy: 0.9928
Epoch 5/5 discriminator_loss: 0.0687 - generator_loss: 9.5255 - loss: 2.9274 - discriminator_accuracy: 0.9906
```
- `python3 gan.py --dataset=mnist-fashion --z_dim=100 --epochs=5`
```
Epoch 1/5 discriminator_loss: 0.0710 - generator_loss: 7.7963 - loss: 1.8059 - discriminator_accuracy: 0.9803
Epoch 2/5 discriminator_loss: 0.0728 - generator_loss: 7.2306 - loss: 2.4866 - discriminator_accuracy: 0.9910
Epoch 3/5 discriminator_loss: 0.1112 - generator_loss: 5.6444 - loss: 1.8976 - discriminator_accuracy: 0.9852
Epoch 4/5 discriminator_loss: 0.1899 - generator_loss: 4.5056 - loss: 1.6542 - discriminator_accuracy: 0.9748
Epoch 5/5 discriminator_loss: 0.3114 - generator_loss: 4.0829 - loss: 1.5674 - discriminator_accuracy: 0.9381
```
- `python3 gan.py --dataset=mnist-cifarcars --z_dim=2 --epochs=5`
```
Epoch 1/5 discriminator_loss: 0.7178 - generator_loss: 4.3867 - loss: 0.9027 - discriminator_accuracy: 0.8721
Epoch 2/5 discriminator_loss: 0.3499 - generator_loss: 4.4815 - loss: 2.1730 - discriminator_accuracy: 0.9631
Epoch 3/5 discriminator_loss: 0.7672 - generator_loss: 2.7376 - loss: 1.2015 - discriminator_accuracy: 0.8301
Epoch 4/5 discriminator_loss: 0.6904 - generator_loss: 2.9754 - loss: 1.2297 - discriminator_accuracy: 0.8599
Epoch 5/5 discriminator_loss: 0.8773 - generator_loss: 2.4737 - loss: 1.1036 - discriminator_accuracy: 0.7979
```
- `python3 gan.py --dataset=mnist-cifarcars --z_dim=100 --epochs=5`
```
Epoch 1/5 discriminator_loss: 0.5299 - generator_loss: 4.1585 - loss: 1.2538 - discriminator_accuracy: 0.8787
Epoch 2/5 discriminator_loss: 0.6910 - generator_loss: 2.3183 - loss: 0.9271 - discriminator_accuracy: 0.8682
Epoch 3/5 discriminator_loss: 1.1221 - generator_loss: 1.9830 - loss: 1.1333 - discriminator_accuracy: 0.7479
Epoch 4/5 discriminator_loss: 1.3696 - generator_loss: 1.0735 - loss: 0.8271 - discriminator_accuracy: 0.6637
Epoch 5/5 discriminator_loss: 1.4549 - generator_loss: 0.9048 - loss: 0.7935 - discriminator_accuracy: 0.5939
```
#### Examples End:
