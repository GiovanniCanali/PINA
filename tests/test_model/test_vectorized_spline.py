import torch
import pytest
import random
from pina.model import VectorizedSpline, Spline
from pina.operator import grad
from pina import LabelTensor

# Utility quantities for testing
order = torch.randint(3, 6, (1,)).item()
n_ctrl_pts = torch.randint(order, order + 5, (1,)).item()
n_knots = order + n_ctrl_pts
n_splines = torch.randint(2, 5, (1,)).item()
output_dim = torch.randint(1, 4, (1,)).item()

# Input points
labels = [f"x{i}" for i in range(n_splines)]
pts = torch.rand(10, n_splines).requires_grad_(True)
pts = LabelTensor(pts, labels)


# Define all possible combinations of valid arguments for VectorizedSpline class
valid_args = [
    {
        "order": order,
        "control_points": torch.rand(n_splines, output_dim, n_ctrl_pts),
        "knots": torch.linspace(0, 1, n_knots)
        .unsqueeze(0)
        .repeat(n_splines, 1),
    },
    {
        "order": order,
        "control_points": torch.rand(n_splines, output_dim, n_ctrl_pts),
        "knots": {
            "n": n_knots,
            "min": 0,
            "max": 1,
            "mode": "auto",
            "n_splines": n_splines,
        },
    },
    {
        "order": order,
        "control_points": torch.rand(n_splines, output_dim, n_ctrl_pts),
        "knots": {
            "n": n_knots,
            "min": 0,
            "max": 1,
            "mode": "uniform",
            "n_splines": n_splines,
        },
    },
    {
        "order": order,
        "control_points": None,
        "knots": torch.linspace(0, 1, n_knots)
        .unsqueeze(0)
        .repeat(n_splines, 1),
    },
    {
        "order": order,
        "control_points": None,
        "knots": {
            "n": n_knots,
            "min": 0,
            "max": 1,
            "mode": "auto",
            "n_splines": n_splines,
        },
    },
    {
        "order": order,
        "control_points": None,
        "knots": {
            "n": n_knots,
            "min": 0,
            "max": 1,
            "mode": "uniform",
            "n_splines": n_splines,
        },
    },
    {
        "order": order,
        "control_points": torch.rand(n_splines, output_dim, n_ctrl_pts),
        "knots": None,
    },
]


@pytest.mark.parametrize("args", valid_args)
@pytest.mark.parametrize("aggregate_output", ["mean", "sum", None])
def test_constructor(args, aggregate_output):
    VectorizedSpline(**args, aggregate_output=aggregate_output)

    # Should fail if order is not a positive integer
    with pytest.raises(AssertionError):
        VectorizedSpline(
            order=-1,
            control_points=args["control_points"],
            knots=args["knots"],
            aggregate_output=aggregate_output,
        )

    # Should fail if control_points is not None or a torch.Tensor
    with pytest.raises(ValueError):
        VectorizedSpline(
            order=args["order"],
            control_points=[1, 2, 3],
            knots=args["knots"],
            aggregate_output=aggregate_output,
        )

    # Should fail if knots is not None, a torch.Tensor, or a dict
    with pytest.raises(ValueError):
        VectorizedSpline(
            order=args["order"],
            control_points=args["control_points"],
            knots=5,
            aggregate_output=aggregate_output,
        )

    # Should fail if aggregate_output is not None, "mean", or "sum"
    with pytest.raises(ValueError):
        VectorizedSpline(
            order=args["order"],
            control_points=args["control_points"],
            knots=args["knots"],
            aggregate_output="invalid",
        )

    # Should fail if both knots and control_points are None
    with pytest.raises(ValueError):
        VectorizedSpline(
            order=args["order"],
            control_points=None,
            knots=None,
            aggregate_output=aggregate_output,
        )

    # Should fail if knots is not two-dimensional
    with pytest.raises(ValueError):
        VectorizedSpline(
            order=args["order"],
            control_points=args["control_points"],
            knots=torch.rand(n_knots),
            aggregate_output=aggregate_output,
        )

    # Should fail if control_points is not three-dimensional
    with pytest.raises(ValueError):
        VectorizedSpline(
            order=args["order"],
            control_points=torch.rand(n_ctrl_pts),
            knots=args["knots"],
            aggregate_output=aggregate_output,
        )

    # Should fail if the number of knots != order + number of control points
    # If control points are None, they are initialized to fulfill this condition
    if args["control_points"] is not None:
        with pytest.raises(ValueError):
            VectorizedSpline(
                order=args["order"],
                control_points=args["control_points"],
                knots=torch.linspace(0, 1, n_knots + 1)
                .unsqueeze(0)
                .repeat(n_splines, 1),
                aggregate_output=aggregate_output,
            )

    # Should fail if the knot dict is missing required keys
    with pytest.raises(ValueError):
        VectorizedSpline(
            order=args["order"],
            control_points=args["control_points"],
            knots={"n": n_knots, "min": 0, "max": 1},
            aggregate_output=aggregate_output,
        )

    # Should fail if the knot dict has invalid 'mode' key
    with pytest.raises(ValueError):
        VectorizedSpline(
            order=args["order"],
            control_points=args["control_points"],
            knots={"n": n_knots, "min": 0, "max": 1, "mode": "invalid"},
            aggregate_output=aggregate_output,
        )

    # Should fail if knots and control points have different number of splines
    with pytest.raises(ValueError):
        VectorizedSpline(
            order=3,
            control_points=torch.rand(5, 4, 5),
            knots=torch.linspace(0, 1, 8).unsqueeze(0).repeat(3, 1),
            aggregate_output=aggregate_output,
        )


@pytest.mark.parametrize("args", valid_args)
def test_forward(args):

    # Define the model
    model = VectorizedSpline(**args)

    # Evaluate the model
    output_ = model(pts)

    # Check output shape
    if model.aggregate_output is None:
        assert output_.shape == (
            pts.shape[0],
            pts.shape[1],
            model.control_points.shape[1],
        )
    else:
        assert output_.shape == pts.shape


@pytest.mark.parametrize("args", valid_args)
def test_backward(args):

    # Define the model
    model = VectorizedSpline(**args)

    # Evaluate the model
    output_ = model(pts)
    loss = torch.mean(output_)
    loss.backward()
    assert model.control_points.grad.shape == model.control_points.shape


@pytest.mark.parametrize("args", valid_args)
def test_derivative(args):

    # Define and evaluate the model
    model = VectorizedSpline(**args)
    pts.requires_grad_(True)
    output_ = model(pts)

    # Compute analytical derivatives
    first_der = model.derivative(x=pts, degree=1)

    # Compute autograd derivatives -- we need to loop over output dimensions
    # since autograd doesn't support vectorized outputs
    gradients = []
    for j in range(output_.shape[2]):
        out = output_[:, :, j].squeeze(-1)
        out = LabelTensor(out, [f"u{j}" for j in range(out.shape[1])])
        gradients.append(
            grad(out, pts)[[f"du{j}dx{j}" for j in range(pts.shape[1])]]
        )
    first_der_auto = torch.stack(gradients, dim=-1)

    # Check shape and value
    assert first_der.shape == first_der_auto.shape
    assert torch.allclose(first_der, first_der_auto, atol=1e-4, rtol=1e-4)


def test_1d_vs_vectorized():

    control_points = torch.rand(1, 1, n_ctrl_pts)
    knots = torch.linspace(0, 1, n_knots).unsqueeze(0)

    # Classical 1D spline

    spline = Spline(
        order=order,
        control_points=control_points.squeeze(),
        knots=knots.squeeze(),
    )

    # Create a VectorizedSpline instance with the same control pts and knots
    vectorized_spline = VectorizedSpline(
        order=order,
        knots=knots,
        control_points=control_points,
        aggregate_output=None,
    )

    # Input points
    x = LabelTensor(torch.rand(10, 1), labels=["x"])

    # Evaluate both models on the same input
    out_spline = spline(x)
    out_vectorized = vectorized_spline(x)

    assert out_vectorized.shape == out_spline.shape
    assert torch.allclose(out_vectorized, out_spline, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("args", valid_args)
@pytest.mark.parametrize("aggregate_output", ["mean", "sum", None])
@pytest.mark.parametrize("insertion_ndim", [1, 2])
def test_refine(args, aggregate_output, insertion_ndim):

    # Define the model and use double precision for geometry comparisons
    model = VectorizedSpline(**args, aggregate_output=aggregate_output).double()

    # Store the knots, control points, and order for comparison
    knots = model.knots.clone()
    ctrl = model.control_points.detach().clone()
    order = model.order

    # Define the number of splines and new knots to insert
    n_splines, n_new = knots.shape[0], random.randint(2, 4)

    def find_new_knots(knot_vector, count):
        """
        Generate `count` new knot values that lie strictly inside the interval
        defined by the first and last elements of `knot_vector`.
        """
        span = knot_vector[1:] - knot_vector[:-1]
        idx = span.argmax()
        left = knot_vector[idx]
        right = knot_vector[idx + 1]
        assert right > left

        # Strictly interior, equally spaced values
        fractions = torch.arange(
            1,
            count + 1,
            dtype=knot_vector.dtype,
            device=knot_vector.device,
        ) / (count + 1)
        return left + fractions * (right - left)

    # Shared insertion values
    if insertion_ndim == 1:

        # Find bounds for the common interior span across all splines
        left, right = knots[:, 0].max(), knots[:, -1].min()

        # Using the union avoids existing knots in any spline
        common = knots[(knots >= left) & (knots <= right)].unique(sorted=True)

        # Find new knots that lie strictly inside the common span
        inserted = find_new_knots(common, n_new)

    # Different insertion values for each spline
    else:
        inserted = torch.stack([find_new_knots(row, n_new) for row in knots])

    # Sample the full knot interval, including both boundaries
    t = torch.linspace(0, 1, 201, dtype=knots.dtype).unsqueeze(1)
    x = knots[:, 0] + t * (knots[:, -1] - knots[:, 0])

    # Evaluate the model's output before and after refinement
    with torch.no_grad():
        before = model(x)
        model.refine(inserted)
        after = model(x)

    # Order and output shape remain unchanged
    assert model.order == order
    assert after.shape == before.shape

    # Each insertion adds one knot and one control point per spline
    assert model.knots.shape == (n_splines, knots.shape[-1] + n_new)
    assert model.control_points.shape == (
        n_splines,
        ctrl.shape[1],
        ctrl.shape[-1] + n_new,
    )

    # The refined knot vectors contain exactly the requested insertions
    if insertion_ndim == 1:
        inserted = inserted.unsqueeze(0).expand(n_splines, -1)
    expected_knots = torch.cat((knots, inserted), dim=-1).sort(dim=-1).values
    torch.testing.assert_close(model.knots, expected_knots)

    # Refinement preserves the represented geometry
    torch.testing.assert_close(after, before, atol=1e-10, rtol=1e-10)
