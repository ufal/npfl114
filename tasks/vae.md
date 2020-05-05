### Assignment: vae
#### Date: Deadline: May 17, 23:59
#### Points: 3 points
#### Examples: vae_example

In this assignment you will implement a simple Variational Autoencoder
for three datasets in the MNIST format. Your goal is to modify the
[vae.py](https://github.com/ufal/npfl114/tree/master/labs/10/vae.py)
template and implement a VAE.

After submitting the assignment to ReCodEx, you can experiment with the three
available datasets (`mnist`, `mnist-fashion`, and `mnist-cifarcars`) and
different latent variable dimensionality (`z_dim=2` and `z_dim=100`).
The generated images are available in TensorBoard logs.

#### Examples Start: vae_example
_Note that the results might be slightly different, depending on your CPU type and whether you use GPU._

- `python3 vae.py --recodex --seed=7 --batch_size=50 --dataset=mnist-recodex --decoder_layers=500,500 --encoder_layers=500,500 --epochs=2 --threads=1 --z_dim=2`
  ```
  2357.67
  ```
- `python3 vae.py --recodex --seed=7 --batch_size=50 --dataset=mnist-recodex --decoder_layers=500,500 --encoder_layers=500,500 --epochs=2 --threads=1 --z_dim=100`
  ```
  2174.10
  ```
#### Examples End:
