"""Module for the B-Spline model class."""

import torch
from pina._src.core.utils import check_consistency
from .vectorized_spline import VectorizedSpline


class Spline(VectorizedSpline):
    r"""
    The univariate B-Spline curve model class.

    A univariate B-spline curve of order :math:`k` is a parametric curve defined
    as a linear combination of B-spline basis functions and control points:

    .. math::

        S(x) = \sum_{i=1}^{n} B_{i,k}(x) C_i, \quad x \in [x_1, x_m]

    where:

    - :math:`C \in \mathbb{R}^n` are the learnable control coefficients. Its
      entries :math:`C_i` influence the shape of the curve but are not generally
      interpolated, except under certain knot multiplicities.
    - :math:`B_{i,k}(x)` are the B-spline basis functions of order :math:`k`,
      i.e., piecewise polynomials of degree :math:`k-1` with support on the
      interval :math:`[x_i, x_{i+k}]`.
    - :math:`X = \{ x_1, x_2, \dots, x_m \}` is the non-decreasing knot vector.

    If the first and last knots are repeated :math:`k` times, then the curve
    interpolates the first and last control coefficients.

    .. note::

        The curve is forced to be zero outside the interval defined by the
        first and last knots.

    :Example:

    >>> from pina.model import Spline
    >>> import torch

    >>> knots1 = torch.tensor([0.0, 0.0, 0.0, 1.0, 2.0, 2.0, 2.0])
    >>> spline1 = Spline(order=3, knots=knots1, control_points=None)

    >>> knots2 = {"n": 7, "min": 0.0, "max": 2.0, "mode": "auto"}
    >>> spline2 = Spline(order=3, knots=knots2, control_points=None)

    >>> knots3 = torch.tensor([0.0, 0.0, 0.0, 1.0, 2.0, 2.0, 2.0])
    >>> control_points3 = torch.tensor([0.0, 1.0, 3.0, 2.0])
    >>> spline3 = Spline(order=3, knots=knots3, control_points=control_points3)
    """

    def __init__(self, order=4, knots=None, control_points=None):
        """
        Initialization of the :class:`Spline` class.

        :param int order: The order of the spline. The corresponding basis
            functions are polynomials of degree ``order - 1``. Default is 4.
        :param knots: The knots of the spline. If a tensor is provided, knots
            are set directly from the tensor. If a dictionary is provided, it
            must contain the keys ``"n"``, ``"min"``, ``"max"``, and ``"mode"``.
            Here, ``"n"`` specifies the number of knots, ``"min"`` and ``"max"``
            define the interval, and ``"mode"`` selects the sampling strategy.
            The supported modes are ``"uniform"``, where the knots are evenly
            spaced over :math:`[min, max]`, and ``"auto"``, where knots are
            constructed to ensure that the spline interpolates the first and
            last control points. In this case, the number of knots is adjusted
            if :math:`n < 2 * order`. If None is given, knots are initialized
            automatically over :math:`[0, 1]` ensuring interpolation of the
            first and last control points. Default is None.
        :type knots: torch.Tensor | dict
        :param torch.Tensor control_points: The control points of the spline.
            If None, they are initialized as learnable parameters with an
            initial value of zero. Default is None.
        :raises AssertionError: If ``order`` is not a positive integer.
        :raises ValueError: If ``knots`` is neither a torch.Tensor nor a
            dictionary, when provided.
        :raises ValueError: If ``control_points`` is not a torch.Tensor,
            when provided.
        :raises ValueError: If both ``knots`` and ``control_points`` are None.
        :raises ValueError: If ``knots`` is neither one-dimensional nor a
            two-dimensional tensor containing a single knot vector.
        :raises ValueError: If ``control_points`` is neither one-dimensional
            nor an expanded single-spline, scalar-output tensor.
        :raises ValueError: If the number of ``knots`` is not equal to the sum
            of ``order`` and the number of ``control_points``.
        :raises UserWarning: If the number of control points is lower than the
            order, resulting in a degenerate spline.
        """
        # Unsqueeze control points if they are one-dimensional for consistency
        if (
            knots is None
            and isinstance(control_points, torch.Tensor)
            and control_points.ndim == 1
        ):
            control_points = control_points.unsqueeze(0)

        # Parent class initialization
        super().__init__(
            order=order, knots=knots, control_points=control_points
        )

    @VectorizedSpline.knots.setter
    def knots(self, value):
        """
        Set the knots of the single univariate spline.

        :param value: A tensor with shape ``[n]`` or ``[1, n]``, or a knot
            dictionary containing ``"n"``, ``"min"``, ``"max"``, and ``"mode"``.
            The dictionary is copied and ``"n_splines"`` is set to ``1``.
        :type value: torch.Tensor | dict
        :raises ValueError: If the value does not describe a single knot vector.
        """
        # Check that the value describes a single knot vector
        if isinstance(value, torch.Tensor):
            if value.ndim == 1:
                value = value.unsqueeze(0)
            elif value.ndim != 2 or value.shape[0] != 1:
                raise ValueError("knots must have shape [n] or [1, n].")
        else:
            value = {**value, "n_splines": 1}

        # Set the knots using the parent class's setter
        VectorizedSpline.knots.fset(self, value)

    @VectorizedSpline.control_points.setter
    def control_points(self, value):
        """
        Set the control points of the single scalar-valued spline.

        :param torch.Tensor value: A tensor with shape ``[c]``, ``[1, c]``, or
            ``[1, 1, c]``. If None, control points are initialized as learnable
            parameters with zero initial value.
        :type value: torch.Tensor | None
        :raises ValueError: If the tensor does not describe one scalar spline.
        """
        # Check that the value describes a single scalar spline
        if value is not None:
            if value.ndim == 1:
                value = value.unsqueeze(0)
            elif not (
                (value.ndim == 2 and value.shape[0] == 1)
                or (value.ndim == 3 and value.shape[:2] == (1, 1))
            ):
                raise ValueError(
                    "control_points must have shape [c], [1, c], or [1, 1, c]."
                )

        # Set the control points using the parent class's setter
        VectorizedSpline.control_points.fset(self, value)
