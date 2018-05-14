_This is an experimental task which might require a lot of time to solve._

The goal of this assignment is to extend the `reinforce_with_baseline`
assignment to make it work on pixel inputs.

The supplied [cart_pole_pixels_evaluator.py](https://github.com/ufal/npfl114/tree/master/labs/12/cart_pole_pixels_evaluator.py)
module (depending on [gym_evaluator.py](https://github.com/ufal/npfl114/tree/master/labs/11/gym_evaluator.py)
generates a pixel representation of the `CartPole` environment
as 80×80 image with three channels, with each channel representing one time step
(i.e., the current situation and the two previous ones).

Start with the [reinforce_with_pixels.py](https://github.com/ufal/npfl114/tree/master/labs/12/reinforce_with_pixels.py)
template, which contains a rich collection of summaries that you can use to
explore the behaviour of the model being trained.

Note that this assignment is not trivial – it takes some time and resources to
make the training progress at all. To show any progress, your goal is to
reach an average reward of 50 during 100 evaluation episodes. As before, the
evaluation period begins only after you call `reset` with `start_evaluate`.

During evaluation in ReCodEx, two different random seeds will be employed, and
you will get a point for each setting where you reach the required reward.
The time limit for each test is 10 minutes.

Because the time limit is 10 minutes per episode, you cannot probably train
the model directly in ReCodEx. Instead, you need to save the trained model and embed
it in your Python solution (see the `gym_cartpole` assignment for an example
of saving the model and then embedding it in a Python source).
