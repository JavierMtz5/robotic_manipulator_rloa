from typing import Optional, Tuple, Any

import torch
from torch import nn
from torch.distributions import MultivariateNormal


class NAF(nn.Module):

    def __init__(self, state_size: int, action_size: int, layer_size: int, seed: int, device: torch.device) -> None:
        """
        Model to be used in the NAF algorithm. Network Architecture:\n
        - Common network\n
            - Linear + BatchNormalization (input_shape, layer_size)\n
            - Linear + BatchNormalization (layer_size, layer_size)\n

        - Output for mu network (used for calculating A)\n
            - Linear (layer_size, action_size)\n

        - Output for V network (used for calculating Q = A + V)\n
            - Linear (layer_size, 1)\n

        - Output for L network (used for calculating P = L . Lt)\n
            - Linear (layer_size, (action_size*action_size+1)/2)\n
        Args:
            state_size: Dimension of a state.
            action_size: Dimension of an action.
            layer_size: Size of the hidden layers of the neural network.
            seed: Random seed.
            device: CUDA device.
        """
        super(NAF, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.state_size = state_size
        self.action_size = action_size
        self.device = device

        # DEFINE THE MODEL

        # Define the first NN hidden layer + BatchNormalization
        self.input_layer = nn.Linear(in_features=self.state_size, out_features=layer_size)
        self.bn1 = nn.BatchNorm1d(layer_size)

        # Define the second NN hidden layer + BatchNormalization
        self.hidden_layer = nn.Linear(in_features=layer_size, out_features=layer_size)
        self.bn2 = nn.BatchNorm1d(layer_size)

        # Define the output layer for the mu Network
        self.action_values = nn.Linear(in_features=layer_size, out_features=action_size)
        # Define the output layer for the V Network
        self.value = nn.Linear(in_features=layer_size, out_features=1)
        # Define the output layer for the L Network
        self.matrix_entries = nn.Linear(in_features=layer_size,
                                        out_features=int(self.action_size * (self.action_size + 1) / 2))

    def forward(self,
                input_: torch.Tensor,
                action: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[Any], Any]:
        """
        Forward propagation.
        It feeds the NN with the input, and gets the output for the mu, V and L networks.\n
        - Output from the L network is used to create the P matrix.\n
        - Output from the V network is used to calculate the Q value: Q = A + V\n
        - Output from the mu network is used to calculate A. The action output of mu nn is considered
            the action that maximizes Q-function.
        Args:
            input_: Input for the neural network's input layer.
            action: Current action, used for calculating the Q-Function estimate.
        Returns:
            Returns a tuple containing the action which maximizes the Q-Function, the
            Q-Function estimate and the Value Function.
        """
        # ============ FEED INPUT DATA TO THE NEURAL NETWORK =================================

        # Feed the input to the INPUT_LAYER and apply ReLu activation function (+ BatchNorm)
        x = torch.relu(self.bn1(self.input_layer(input_)))
        # Feed the output of INPUT_LAYER to the HIDDEN_LAYER layer and apply ReLu activation function (+ BatchNorm)
        x = torch.relu(self.bn2(self.hidden_layer(x)))

        # Feed the output of HIDDEN_LAYER to the mu layer and apply tanh activation function
        action_value = torch.tanh(self.action_values(x))

        # Feed the output of HIDDEN_LAYER to the L layer and apply tanh activation function
        matrix_entries = torch.tanh(self.matrix_entries(x))

        # Feed the output of HIDDEN_LAYER to the V layer
        V = self.value(x)

        # Modifies the output of the mu layer by unsqueezing it (all tensor as a 1D vector)
        action_value = action_value.unsqueeze(-1)

        # ============ CREATE L MATRIX from the outputs of the L layer =======================

        # Create lower-triangular matrix, size: (n_samples, action_size, action_size)
        L = torch.zeros((input_.shape[0], self.action_size, self.action_size)).to(self.device)
        # Get lower triagular indices (returns list of 2 elems, where the first row contains row coordinates
        # of all indices and the second row contains column coordinates)
        lower_tri_indices = torch.tril_indices(row=self.action_size, col=self.action_size, offset=0)
        # Fill matrix with the outputs of the L layer
        L[:, lower_tri_indices[0], lower_tri_indices[1]] = matrix_entries
        # Raise the diagonal elements of the matrix to the square
        L.diagonal(dim1=1, dim2=2).exp_()
        # Calculate state-dependent, positive-definite square matrix P
        P = L * L.transpose(2, 1)

        # ============================ CALCULATE Q-VALUE ===================================== #

        Q = None
        if action is not None:
            # Calculate Advantage Function estimate
            A = (-0.5 * torch.matmul(torch.matmul((action.unsqueeze(-1) - action_value).transpose(2, 1), P),
                                     (action.unsqueeze(-1) - action_value))).squeeze(-1)

            # Calculate Q-values
            Q = A + V

        # =========================== ADD NOISE TO ACTION ==================================== #

        dist = MultivariateNormal(action_value.squeeze(-1), torch.inverse(P))
        action = dist.sample()
        action = torch.clamp(action, min=-1, max=1)

        return action, Q, V
