import torch
import random
import pytest
from pina.model import SplineSurface
from pina.operator import grad
from pina import LabelTensor

# Utility quantities for testing
orders = [random.randint(3, 6) for _ in range(2)]
n_ctrl_pts = random.randint(max(orders), max(orders) + 5)
n_knots = [orders[i] + n_ctrl_pts for i in range(2)]

# Input tensor
points = [
    LabelTensor(torch.rand(100, 2), ["x", "y"]),
    LabelTensor(torch.rand(2, 100, 2), ["x", "y"]),
]


@pytest.mark.parametrize(
    "knots_u",
    [
        torch.rand(n_knots[0]),
        {"n": n_knots[0], "min": 0, "max": 1, "mode": "auto"},
        {"n": n_knots[0], "min": 0, "max": 1, "mode": "uniform"},
        None,
    ],
)
@pytest.mark.parametrize(
    "knots_v",
    [
        torch.rand(n_knots[1]),
        {"n": n_knots[1], "min": 0, "max": 1, "mode": "auto"},
        {"n": n_knots[1], "min": 0, "max": 1, "mode": "uniform"},
        None,
    ],
)
@pytest.mark.parametrize(
    "control_points", [torch.rand(n_ctrl_pts, n_ctrl_pts), None]
)
def test_constructor(knots_u, knots_v, control_points):

    # Skip if knots_u, knots_v, and control_points are all None
    if (knots_u is None or knots_v is None) and control_points is None:
        return

    SplineSurface(
        orders=orders,
        knots_u=knots_u,
        knots_v=knots_v,
        control_points=control_points,
    )

    # Should fail if orders is not list of two elements
    with pytest.raises(ValueError):
        SplineSurface(
            orders=[orders[0]],
            knots_u=knots_u,
            knots_v=knots_v,
            control_points=control_points,
        )

    # Should fail if both knots and control_points are None
    with pytest.raises(ValueError):
        SplineSurface(
            orders=orders,
            knots_u=None,
            knots_v=None,
            control_points=None,
        )

    # Should fail if control_points is not a torch.Tensor when provided
    with pytest.raises(ValueError):
        SplineSurface(
            orders=orders,
            knots_u=knots_u,
            knots_v=knots_v,
            control_points=[[0.0] * n_ctrl_pts] * n_ctrl_pts,
        )

    # Should fail if control_points is not of the correct shape when provided
    # It assumes that at least one among knots_u and knots_v is not None
    if knots_u is not None or knots_v is not None:
        with pytest.raises(ValueError):
            SplineSurface(
                orders=orders,
                knots_u=knots_u,
                knots_v=knots_v,
                control_points=torch.rand(n_ctrl_pts + 1, n_ctrl_pts + 1),
            )

    # Should fail if there are not enough knots_u to define the control points
    with pytest.raises(ValueError):
        SplineSurface(
            orders=orders,
            knots_u=torch.linspace(0, 1, orders[0]),
            knots_v=knots_v,
            control_points=None,
        )

    # Should fail if there are not enough knots_v to define the control points
    with pytest.raises(ValueError):
        SplineSurface(
            orders=orders,
            knots_u=knots_u,
            knots_v=torch.linspace(0, 1, orders[1]),
            control_points=None,
        )


@pytest.mark.parametrize(
    "knots_u",
    [
        torch.rand(n_knots[0]),
        {"n": n_knots[0], "min": 0, "max": 1, "mode": "auto"},
        {"n": n_knots[0], "min": 0, "max": 1, "mode": "uniform"},
    ],
)
@pytest.mark.parametrize(
    "knots_v",
    [
        torch.rand(n_knots[1]),
        {"n": n_knots[1], "min": 0, "max": 1, "mode": "auto"},
        {"n": n_knots[1], "min": 0, "max": 1, "mode": "uniform"},
    ],
)
@pytest.mark.parametrize(
    "control_points", [torch.rand(n_ctrl_pts, n_ctrl_pts), None]
)
@pytest.mark.parametrize("pts", points)
def test_forward(knots_u, knots_v, control_points, pts):

    # Define the model
    model = SplineSurface(
        orders=orders,
        knots_u=knots_u,
        knots_v=knots_v,
        control_points=control_points,
    )

    # Evaluate the model
    output_ = model(pts)
    assert output_.shape == (*pts.shape[:-1], 1)


@pytest.mark.parametrize(
    "knots_u",
    [
        torch.rand(n_knots[0]),
        {"n": n_knots[0], "min": 0, "max": 1, "mode": "auto"},
        {"n": n_knots[0], "min": 0, "max": 1, "mode": "uniform"},
    ],
)
@pytest.mark.parametrize(
    "knots_v",
    [
        torch.rand(n_knots[1]),
        {"n": n_knots[1], "min": 0, "max": 1, "mode": "auto"},
        {"n": n_knots[1], "min": 0, "max": 1, "mode": "uniform"},
    ],
)
@pytest.mark.parametrize(
    "control_points", [torch.rand(n_ctrl_pts, n_ctrl_pts), None]
)
@pytest.mark.parametrize("pts", points)
def test_backward(knots_u, knots_v, control_points, pts):

    # Define the model
    model = SplineSurface(
        orders=orders,
        knots_u=knots_u,
        knots_v=knots_v,
        control_points=control_points,
    )

    # Evaluate the model
    output_ = model(pts)
    loss = torch.mean(output_)
    loss.backward()
    assert model.control_points.grad.shape == model.control_points.shape


@pytest.mark.parametrize(
    "knots_u",
    [
        torch.rand(n_knots[0]),
        {"n": n_knots[0], "min": 0, "max": 1, "mode": "auto"},
        {"n": n_knots[0], "min": 0, "max": 1, "mode": "uniform"},
    ],
)
@pytest.mark.parametrize(
    "knots_v",
    [
        torch.rand(n_knots[1]),
        {"n": n_knots[1], "min": 0, "max": 1, "mode": "auto"},
        {"n": n_knots[1], "min": 0, "max": 1, "mode": "uniform"},
    ],
)
@pytest.mark.parametrize(
    "control_points", [torch.rand(n_ctrl_pts, n_ctrl_pts), None]
)
@pytest.mark.parametrize("pts", points)
def test_derivative(knots_u, knots_v, control_points, pts):

    # Define and evaluate the model
    model = SplineSurface(
        orders=orders,
        knots_u=knots_u,
        knots_v=knots_v,
        control_points=control_points,
    )
    pts.requires_grad_(True)
    output_ = LabelTensor(model(pts), "u")

    # Compute derivatives
    gradient = model.gradient(x=pts)
    gradient_auto = grad(output_, pts).tensor

    # Check shape and value
    assert gradient.shape == pts.shape
    assert torch.allclose(gradient, gradient_auto, atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize(
    "knots_u",
    [
        torch.rand(n_knots[0]),
        {"n": n_knots[0], "min": 0, "max": 1, "mode": "auto"},
        {"n": n_knots[0], "min": 0, "max": 1, "mode": "uniform"},
    ],
)
@pytest.mark.parametrize(
    "knots_v",
    [
        torch.rand(n_knots[1]),
        {"n": n_knots[1], "min": 0, "max": 1, "mode": "auto"},
        {"n": n_knots[1], "min": 0, "max": 1, "mode": "uniform"},
    ],
)
@pytest.mark.parametrize(
    "control_points", [torch.rand(n_ctrl_pts, n_ctrl_pts), None]
)
@pytest.mark.parametrize("axis", ["u", "v"])
def test_refine(knots_u, knots_v, control_points, axis):

    # Define the model and use double precision for geometry comparisons
    model = SplineSurface(
        orders=orders,
        knots_u=knots_u,
        knots_v=knots_v,
        control_points=control_points,
    ).double()

    # Store both knot vectors, the control points, and orders
    knots_u = model.spline_u.knots.clone()
    knots_v = model.spline_v.knots.clone()
    ctrl = model.control_points.detach().clone()

    # Generate new values inside the widest nonzero span
    n_new = random.randint(2, 4)
    knots = knots_u[0] if axis == "u" else knots_v[0]
    spans = knots[1:] - knots[:-1]
    idx = spans.argmax()

    # Generate new values that lie strictly inside the selected span
    fractions = torch.arange(1, n_new + 1, dtype=knots.dtype) / (n_new + 1)
    inserted = knots[idx] + fractions * (knots[idx + 1] - knots[idx])

    # Sample the full surface parameter domain, including boundaries
    t = torch.linspace(0, 1, 51, dtype=knots.dtype)
    u = knots_u[0, 0] + t * (knots_u[0, -1] - knots_u[0, 0])
    v = knots_v[0, 0] + t * (knots_v[0, -1] - knots_v[0, 0])
    x = torch.cartesian_prod(u, v)

    # Evaluate the model's output before and after refinement
    with torch.no_grad():
        before = model(x)
        model.refine(inserted, axis=axis)
        after = model(x)

    # Orders and output shape remain unchanged
    assert (model.spline_u.order, model.spline_v.order) == tuple(orders)
    assert before.shape == after.shape == (x.shape[0], 1)

    # Only the selected axis gains knots and control points
    added_u = n_new if axis == "u" else 0
    added_v = n_new if axis == "v" else 0

    # Verify that the number of knots and control points has increased correctly
    assert model.spline_u.knots.shape == (1, knots_u.shape[-1] + added_u)
    assert model.spline_v.knots.shape == (1, knots_v.shape[-1] + added_v)
    assert model.control_points.shape == (
        ctrl.shape[0] + added_u,
        ctrl.shape[1] + added_v,
    )

    # Compute expected knot vectors after refinement
    if axis == "u":
        expected_u = torch.cat((knots_u, inserted.unsqueeze(0)), dim=-1)
        expected_u = expected_u.sort(dim=-1).values
        expected_v = knots_v
    else:
        expected_u = knots_u
        expected_v = torch.cat((knots_v, inserted.unsqueeze(0)), dim=-1)
        expected_v = expected_v.sort(dim=-1).values

    # Verify that the knot vectors match the expected values
    torch.testing.assert_close(model.spline_u.knots, expected_u)
    torch.testing.assert_close(model.spline_v.knots, expected_v)

    # Refinement preserves the represented surface geometry
    torch.testing.assert_close(after, before, atol=1e-10, rtol=1e-10)
