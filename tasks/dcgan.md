### Assignment: dcgan
#### Date: Deadline: May 17, 23:59
#### Points: 1 points
#### Examples: dcgan_example

This task is a continuation of the `gan` assignment, which you will modify to
implement the Deep Convolutional GAN (DCGAN).

Start with the
[dcgan.py](https://github.com/ufal/npfl114/tree/master/labs/10/dcgan.py)
template and implement a DCGAN. Note that most of the TODO notes are from
the `gan` assignment.

After submitting the assignment to ReCodEx, you can experiment with the three
available datasets (`mnist`, `mnist-fashion`, and `mnist-cifarcars`). However,
note that you will need _a lot_ of computational power (preferably a GPU) to
generate the images.

#### Examples Start: dcgan_example
_Note that the results might be slightly different, depending on your CPU type and whether you use GPU._

- `python3 dcgan.py --recodex --seed=7 --batch_size=50 --dataset=mnist-recodex --epochs=1 --threads=1 --z_dim=2`
  ```
  30.34
  ```
- `python3 dcgan.py --recodex --seed=7 --batch_size=50 --dataset=mnist-recodex --epochs=1 --threads=1 --z_dim=100`
  ```
  27.20
  ```
#### Examples End:
