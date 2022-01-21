import torch
import torch.nn
import torch.nn.functional as F


def noop(args):
    return args


class SimpleModel(torch.nn.Module):
    def __init__(self,
                 n_input,
                 n_output,
                 n_hidden,
                 n_hidden_layers=1,
                 activation=torch.nn.ReLU(),
                 output_activation=noop):
        super().__init__()
        self.n_input = n_input
        self.n_output = n_output
        self.n_hidden = n_hidden
        self.n_hidden_layers = n_hidden_layers
        self.activation = activation

        for i in range(self.n_hidden_layers):
            layer = torch.nn.Linear(in_features=self.n_input if i == 0 else self.n_hidden,
                                    out_features=self.n_hidden)
            # He weight initialisation
            torch.nn.init.kaiming_uniform_(layer.weight)
            # Add layer to module
            self.__setattr__("hidden_layer_" + str(i),
                             layer)
        # add output layer to module
        self.output_layer = torch.nn.Linear(in_features=self.n_hidden,
                                            out_features=self.n_output)
        # He weight initialisation
        torch.nn.init.kaiming_uniform_(self.output_layer.weight)

        # Add output activation function
        self.output_activation = output_activation

    def forward(self, X):
        # feed hidden layers
        for i in range(self.n_hidden_layers):
            X = self.__getattr__("hidden_layer_" + str(i))(X)
            X = self.activation(X)
        # feed output layer
        return self.output_activation(self.output_layer(X))


class GAN(torch.nn.Sequential):
    def __init__(self, g, d, *args):
        super().__init__(*args)
        self.add_module("generator", g)
        self.add_module("discriminator", d)
