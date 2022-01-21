# Part 2

In this part, we will look at GANs, where the input isn't restricted to noise.

In particular, this part will look at GANs that have a label as input.

The first experiment will train on the MNIST database, and an extra input to the model, that sets the number.

This variant is a so-called conditional GAN (cGAN).


## Theory

TODO this section should include own interpretation of:
[Mirza and Osindero, 2014, Conditional Generative Adversarial Nets](https://arxiv.org/abs/1411.1784)



## Notebooks

### 01_MNIST

This part is directly applying a conditional to the last model of the previous section (part1). As the input to the
models already take a flat input, it's straightforward to extend the input space by 10 for one-hot encoding.

However, one addition is the conditional parsed to the generator must naturally also parse on to the discriminator.

It follows then, that the latent sample-generator should generate a conditional in addition to the noise seen earlier.
