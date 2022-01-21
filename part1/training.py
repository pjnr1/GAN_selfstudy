from functools import partial

import torch


def train_GAN(generator, discriminator, gan_model, latent_dim,
              real_generator, fake_generator, latent_generator,
              loss_func=torch.nn.MSELoss,
              n_epochs=10000, n_batch=128, n_eval=2000,
              summarize_func=None,
              optimizer=partial(torch.optim.Adam, lr=0.0002)):
    def _train_batch(X, y, model, optim):
        optim.zero_grad()
        loss = loss_func()(model(X), y)
        loss.backward()
        optim.step()
        return model, optim, loss

    # setup optimizers
    optimizer_discriminator = optimizer(discriminator.parameters())
    optimizer_generator = optimizer(generator.parameters())

    # determine half the size of one batch, for updating the discriminator
    half_batch = int(n_batch / 2)

    # for-loop over epochs
    for i in range(n_epochs):
        # prepare real samples
        x_real, y_real = real_generator(half_batch)
        # prepare fake examples
        x_fake, y_fake = fake_generator(generator, latent_dim, half_batch)
        # train discriminator
        discriminator, optimizer_discriminator, loss_real = _train_batch(x_real,
                                                                         y_real,
                                                                         discriminator,
                                                                         optimizer_discriminator)
        discriminator, optimizer_discriminator, loss_fake = _train_batch(x_fake,
                                                                         y_fake,
                                                                         discriminator,
                                                                         optimizer_discriminator)
        # prepare points in latent space as input for the generator
        x_gan = latent_generator(latent_dim, n_batch)
        # create inverted labels for the fake samples
        y_gan = torch.ones((n_batch, 1)).float()
        # train the generator via the discriminator's error
        gan_model, optimizer_generator, loss = _train_batch(x_gan,
                                                            y_gan,
                                                            gan_model,
                                                            optimizer_generator)
        # evaluate the model every n_eval epochs
        if (i + 1) % n_eval == 0 and summarize_func is not None:
            summarize_func(i, generator, discriminator, latent_dim)
