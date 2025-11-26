"""Module for the CoupledNetwork model class."""

import torch
from .block import CoupledNetworkBlock
from ..utils import check_consistency, check_positive_integer


class CoupledNetwork(torch.nn.Module):
    """
    Implementation of the CoupledNetwork architecture.

    The model applies an initial embedding layer, followed by a sequence of
    :class:`~pina.model.block.coupled_network_block.CoupledNetworkBlock` blocks,
    and ends with a linear output layer. Each block permutes the input features,
    splits them into two halves, processes them separately, and then recombines
    them.

    This design aims to reduce spectral bias while preventing vanishing
    gradients.

    See :class:`~pina.model.block.coupled_network_block.CoupledNetworkBlock` for
    details on the block structure.

    .. seealso::

        **Original reference**:
        #TODO: Add reference when available.
    """

    def __init__(
        self,
        input_dimension,
        inner_size,
        output_dimension,
        embedding=None,
        n_layers=3,
        activation=torch.nn.Tanh,
    ):
        """
        Initialization of the :class:`CoupledNetwork` class.

        :param int input_dimension: The number of input features.
        :param int inner_size: The number of hidden units in the coupled block.
        :param int output_dimension: The number of output features.
        :param torch.nn.Module embedding: The embedding module used to transform
            the input into a higher-dimensional feature space. If ``None``, a
            simple linear embedding is used. Default is ``None``.
        :param int n_layers: The number of CoupledNetwork blocks to be stacked.
            Default is 3.
        :param torch.nn.Module activation: The activation function to be used in
            the blocks. Default is :class:`torch.nn.Tanh`.
        :raises ValueError: If `activation` is not a :class:`torch.nn.Module`.
        :raises ValueError: If `embedding` is not a :class:`torch.nn.Module` or
            None.
        :raises AssertionError: If `input_dimension` is not a positive integer.
        :raises AssertionError: If `inner_size` is not a positive integer.
        :raises AssertionError: If `output_dimension` is not a positive integer.
        :raises AssertionError: If `n_layers` is not a positive integer.
        :raises ValueError: If `inner_size` is less than 2.
        """
        super().__init__()

        # Check consistency
        check_consistency(activation, torch.nn.Module, subclass=True)
        check_consistency(embedding, (torch.nn.Module, type(None)))
        check_positive_integer(input_dimension, strict=True)
        check_positive_integer(inner_size, strict=True)
        check_positive_integer(output_dimension, strict=True)
        check_positive_integer(n_layers, strict=True)

        # Check that inner_size is at least 2 to allow splitting in the block
        if inner_size < 2:
            raise ValueError("inner_size must be at least 2 to allow splitting")

        # Initialize the activation function
        self.activation = activation()

        # Initialize the embedding
        self.embedding = embedding or torch.nn.Linear(
            in_features=input_dimension,
            out_features=inner_size,
        )

        # Initialize the coupled blocks
        self.blocks = torch.nn.ModuleList(
            [
                CoupledNetworkBlock(inner_size, activation)
                for _ in range(n_layers)
            ]
        )

        # Initialize the output layer
        self.output_layer = torch.nn.Linear(
            in_features=inner_size,
            out_features=output_dimension,
        )

    def forward(self, x):
        """
        Forward pass of the CoupledNetwork model. It applies the embedding
        layer, processes the result through the stacked blocks, and finishes
        with the output layer.

        :param x: The input tensor.
        :type x: torch.Tensor | LabelTensor
        :return: The output tensor.
        :rtype: torch.Tensor | LabelTensor
        """
        # Apply the embedding
        x = self.embedding(x)

        # Pass through the coupled blocks
        for block in self.blocks:
            x = block(x)

        return self.output_layer(x)
