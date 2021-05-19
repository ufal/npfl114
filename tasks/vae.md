### Assignment: vae
#### Date: Deadline: Jun 30, 23:59
#### Points: 3 points
#### Examples: vae_examples

In this assignment you will implement a simple Variational Autoencoder
for three datasets in the MNIST format. Your goal is to modify the
[vae.py](https://github.com/ufal/npfl114/tree/master/labs/12/vae.py)
template and implement a VAE.

After submitting the assignment to ReCodEx, you can experiment with the three
available datasets (`mnist`, `mnist-fashion`, and `mnist-cifarcars`) and
different latent variable dimensionality (`z_dim=2` and `z_dim=100`).
The generated images are available in TensorBoard logs.

#### Examples Start: vae_examples
_Note that your results may be slightly different, depending on your CPU type and whether you use GPU._
- `python3 vae.py --dataset=mnist --z_dim=2 --epochs=3`
```
Epoch 1/3 reconstruction_loss: 0.2159 - latent_loss: 2.4693 - loss: 174.2038
Epoch 2/3 reconstruction_loss: 0.1928 - latent_loss: 2.7937 - loss: 156.7730
Epoch 3/3 reconstruction_loss: 0.1868 - latent_loss: 2.9350 - loss: 152.3162
```
- `python3 vae.py --dataset=mnist --z_dim=100 --epochs=3`
```
Epoch 1/3 reconstruction_loss: 0.1837 - latent_loss: 0.1378 - loss: 157.7933
Epoch 2/3 reconstruction_loss: 0.1319 - latent_loss: 0.1847 - loss: 121.9125
Epoch 3/3 reconstruction_loss: 0.1209 - latent_loss: 0.1903 - loss: 113.7889
```
- `python3 vae.py --dataset=mnist-fashion --z_dim=2 --epochs=3`
```
Epoch 1/3 reconstruction_loss: 0.3539 - latent_loss: 2.9950 - loss: 283.4177
Epoch 2/3 reconstruction_loss: 0.3324 - latent_loss: 3.0159 - loss: 266.6620
Epoch 3/3 reconstruction_loss: 0.3288 - latent_loss: 3.0269 - loss: 263.8320
```
- `python3 vae.py --dataset=mnist-fashion --z_dim=100 --epochs=3`
```
Epoch 1/3 reconstruction_loss: 0.3400 - latent_loss: 0.1183 - loss: 278.3589
Epoch 2/3 reconstruction_loss: 0.3088 - latent_loss: 0.1061 - loss: 252.7133
Epoch 3/3 reconstruction_loss: 0.3029 - latent_loss: 0.1086 - loss: 248.3083
```
- `python3 vae.py --dataset=mnist-cifarcars --z_dim=2 --epochs=3`
```
Epoch 1/3 reconstruction_loss: 0.6373 - latent_loss: 1.9468 - loss: 503.5290
Epoch 2/3 reconstruction_loss: 0.6307 - latent_loss: 2.0624 - loss: 498.5606
Epoch 3/3 reconstruction_loss: 0.6292 - latent_loss: 2.1156 - loss: 497.5026
```
- `python3 vae.py --dataset=mnist-cifarcars --z_dim=100 --epochs=3`
```
Epoch 1/3 reconstruction_loss: 0.6359 - latent_loss: 0.0577 - loss: 504.3351
Epoch 2/3 reconstruction_loss: 0.6164 - latent_loss: 0.0714 - loss: 490.4035
Epoch 3/3 reconstruction_loss: 0.6097 - latent_loss: 0.0860 - loss: 486.5849
```
#### Examples End:
