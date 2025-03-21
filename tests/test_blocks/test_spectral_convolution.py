from pina.model.block import (
    SpectralConvBlock1D,
    SpectralConvBlock2D,
    SpectralConvBlock3D,
)
import torch

input_numb_fields = 3
output_numb_fields = 4
batch = 5


def test_constructor_1d():
    SpectralConvBlock1D(
        input_numb_fields=input_numb_fields,
        output_numb_fields=output_numb_fields,
        n_modes=5,
    )


def test_forward_1d():
    sconv = SpectralConvBlock1D(
        input_numb_fields=input_numb_fields,
        output_numb_fields=output_numb_fields,
        n_modes=4,
    )
    x = torch.rand(batch, input_numb_fields, 10)
    sconv(x)


def test_backward_1d():
    sconv = SpectralConvBlock1D(
        input_numb_fields=input_numb_fields,
        output_numb_fields=output_numb_fields,
        n_modes=4,
    )
    x = torch.rand(batch, input_numb_fields, 10)
    x.requires_grad = True
    sconv(x)
    l = torch.mean(sconv(x))
    l.backward()
    assert x._grad.shape == torch.Size([5, 3, 10])


def test_constructor_2d():
    SpectralConvBlock2D(
        input_numb_fields=input_numb_fields,
        output_numb_fields=output_numb_fields,
        n_modes=[5, 4],
    )


def test_forward_2d():
    sconv = SpectralConvBlock2D(
        input_numb_fields=input_numb_fields,
        output_numb_fields=output_numb_fields,
        n_modes=[5, 4],
    )
    x = torch.rand(batch, input_numb_fields, 10, 10)
    sconv(x)


def test_backward_2d():
    sconv = SpectralConvBlock2D(
        input_numb_fields=input_numb_fields,
        output_numb_fields=output_numb_fields,
        n_modes=[5, 4],
    )
    x = torch.rand(batch, input_numb_fields, 10, 10)
    x.requires_grad = True
    sconv(x)
    l = torch.mean(sconv(x))
    l.backward()
    assert x._grad.shape == torch.Size([5, 3, 10, 10])


def test_constructor_3d():
    SpectralConvBlock3D(
        input_numb_fields=input_numb_fields,
        output_numb_fields=output_numb_fields,
        n_modes=[5, 4, 4],
    )


def test_forward_3d():
    sconv = SpectralConvBlock3D(
        input_numb_fields=input_numb_fields,
        output_numb_fields=output_numb_fields,
        n_modes=[5, 4, 4],
    )
    x = torch.rand(batch, input_numb_fields, 10, 10, 10)
    sconv(x)


def test_backward_3d():
    sconv = SpectralConvBlock3D(
        input_numb_fields=input_numb_fields,
        output_numb_fields=output_numb_fields,
        n_modes=[5, 4, 4],
    )
    x = torch.rand(batch, input_numb_fields, 10, 10, 10)
    x.requires_grad = True
    sconv(x)
    l = torch.mean(sconv(x))
    l.backward()
    assert x._grad.shape == torch.Size([5, 3, 10, 10, 10])
