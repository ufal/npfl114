### Assignment: gan
#### Date: Deadline: May 17, 23:59
#### Points: 3 points
#### Examples: gan_example

In this assignment you will implement a simple Generative Adversarion Network
for three datasets in the MNIST format. Your goal is to modify the
[gan.py](https://github.com/ufal/npfl114/tree/master/labs/10/gan.py)
template and implement a GAN.

After submitting the assignment to ReCodEx, you can experiment with the three
available datasets (`mnist`, `mnist-fashion`, and `mnist-cifarcars`) and
maybe try different latent variable dimensionality. The generated images are
available in TensorBoard logs.

You can also continue with `dcgan` assignment.

#### Examples Start: gan_example
_Note that the results might be slightly different, depending on your CPU type and whether you use GPU._

- `python3 gan.py --recodex --seed=7 --batch_size=50 --dataset=mnist-recodex --discriminator_layers=128 --generator_layers=128 --epochs=2 --threads=1 --z_dim=2`
  ```
  57.75
  ```
- `python3 gan.py --recodex --seed=7 --batch_size=50 --dataset=mnist-recodex --discriminator_layers=128 --generator_layers=128 --epochs=2 --threads=1 --z_dim=100`
  ```
  49.24
  ```
#### Examples End:
