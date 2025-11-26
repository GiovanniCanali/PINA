"""Module for the CoupledNetwork block class."""

import torch
from ...utils import check_consistency, check_positive_integer


class CoupledNetworkBlock(torch.nn.Module):
    r"""
    The inner block of the CoupledNetwork architecture.

    The block first applies a fixed random permutation to the input and splits
    the result into two parts, :math:`x_1` and :math:`x_2`. These are processed
    separately: :math:`x_1` is left unchanged to produce :math:`y_1`, while
    :math:`y_2` is obtained by combining :math:`x_2` with transformations of
    :math:`x_1`.

    .. math::

        y_1 &= x_1 \\
        y_2 &= S \odot x_2 + T

    Here, :math:`T` is computed using two linear layers with an intermediate
    activation, and :math:`S` is obtained by applying an exponential to a
    similar transformation followed by LayerNorm. The operator :math:`\odot`
    denotes element-wise multiplication.

    .. seealso::

        **Original reference**:
        #TODO: Add reference when available.
    """

    def __init__(self, inner_size, activation):
        """
        Initialization of the :class:`CoupledNetworkBlock` class.

        :param int inner_size: The number of hidden units in the block.
        :param torch.nn.Module activation: The activation function.
        :raises ValueError: If `activation` is not a :class:`torch.nn.Module`.
        :raises AssertionError: If `inner_size` is not a positive integer.
        """
        super().__init__()

        # Check consistency
        check_consistency(activation, torch.nn.Module, subclass=True)
        check_positive_integer(inner_size, strict=True)

        # Activation function
        self.activation = activation()

        # Fixed random permutation
        self.register_buffer("permutation", torch.randperm(inner_size))

        # Dimensions of each half
        first_half_size = inner_size // 2 + inner_size % 2
        second_half_size = inner_size - first_half_size

        # Transformation T
        self.t = torch.nn.Sequential(
            torch.nn.Linear(first_half_size, first_half_size),
            self.activation,
            torch.nn.Linear(first_half_size, first_half_size),
        )

        # Transformation S (exponentiation is applied in the forward)
        self.s = torch.nn.Sequential(
            torch.nn.Linear(first_half_size, second_half_size),
            self.activation,
            torch.nn.Linear(second_half_size, second_half_size),
            self.activation,
            torch.nn.LayerNorm(second_half_size),
        )

    def forward(self, x):
        """
        Forward pass of the CoupledNetwork block. It applies the fixed
        permutation to the input, splits it into two parts, processes each half
        independently, and then recombines them into the final output.

        :param x: The input tensor.
        :type x: torch.Tensor | LabelTensor
        :return: The output tensor.
        :rtype: torch.Tensor | LabelTensor
        """
        # Apply the fixed permutation and split the input
        x = x[..., self.permutation]
        x1, x2 = torch.chunk(x, 2, dim=-1)

        # Compute y2. Notice that y1 is just x1
        y2 = torch.exp(self.s(x1)) * x2 + self.t(x1)

        return torch.cat((x1, y2), dim=1)
