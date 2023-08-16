import torch
import random

from conv2d import CustomConv2d
from conv_transposed2d import CustomConvTransposed2d


def _setup(transpose: bool, in_channels, out_channels, kernel_size, stride, padding, dilation, bias, groups,
           output_padding=None):
    if transpose:
        pytorch_conv = torch.nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels,
                                                kernel_size=kernel_size, stride=stride, padding=padding,
                                                output_padding=output_padding, dilation=dilation, groups=groups,
                                                bias=bias)

        custom_conv = CustomConvTransposed2d(in_channels, out_channels, kernel_size, stride, padding, output_padding,
                                             dilation, groups, bias)
    else:
        pytorch_conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)

        custom_conv = CustomConv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)

    pytorch_conv.weight.data = custom_conv.weight
    pytorch_conv.bias.data = custom_conv.bias

    return pytorch_conv, custom_conv


def _test_forward(pytorch_conv, custom_conv, random_data, epsilon=0.01):
    output_pytorch = pytorch_conv(random_data)
    custom_output = custom_conv(random_data)

    assert output_pytorch.shape == custom_output.shape, \
        f"Output shape ({custom_output.shape}) is not equal to PyTorch output shape ({output_pytorch.shape})"

    mse_loss = torch.nn.functional.mse_loss(custom_output, output_pytorch)

    assert mse_loss.item() < epsilon, f"MSE ({mse_loss.item()}) is not less than epsilon ({epsilon})"


def _test_backward(pytorch_conv, custom_conv, random_data, target, epsilon, iterations=100):
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(pytorch_conv.parameters(), lr=0.01)

    for i in range(iterations):
        # Torch weights update
        pytorch_conv.zero_grad()
        output_pytorch = pytorch_conv(random_data)
        pytorch_loss = criterion(target, output_pytorch)
        pytorch_loss.backward()
        w1 = pytorch_conv.weight.grad.clone()
        optimizer.step()
        output_pytorch = pytorch_conv(random_data)
        pytorch_loss = criterion(target, output_pytorch)

        # Custom weights update
        custom_output = custom_conv(random_data)
        custom_loss = criterion(target, custom_output)
        custom_conv.backward(custom_output)
        custom_output = custom_conv(random_data)
        custom_loss = criterion(target, custom_output)
        w2 = custom_conv.weight.grad.clone()

        assert w1.shape == w2.shape, \
            f"Weight gradients shape ({w2.shape}) is not equal to PyTorch weight gradients shape ({w1.shape})"

        mse_grad_w = torch.nn.functional.mse_loss(w1, w2)
        mse_loss = torch.nn.functional.mse_loss(pytorch_loss, custom_loss)
        print(f"MSE of losses: {mse_loss.item()}")
        assert mse_loss.item() < epsilon, f"MSE ({mse_loss.item()}) is not less than epsilon ({epsilon})"


def test_conv2d(iterations=1):
    in_channels = 3
    out_channels = 2
    kernel_size = 3
    stride = 1
    padding = 0
    dilation = 1
    bias = True
    groups = 1
    epsilon = 1e-13
    samples = 1

    random_data = torch.rand(samples, in_channels, 28, 28)
    target = torch.randn(samples, out_channels, 26, 26)

    pytorch_conv, custom_conv = _setup(False, in_channels, out_channels, kernel_size, stride, padding, dilation, bias,
                                       groups)

    _test_forward(pytorch_conv, custom_conv, random_data)
    _test_backward(pytorch_conv, custom_conv, random_data, target, epsilon=1e-4, iterations=iterations)


def test_conv_transposed2d(iterations=1):
    in_channels = 3
    out_channels = 16
    kernel_size = 2
    stride = 2
    padding = 2
    output_padding = 0
    groups = 1
    dilation = 1
    bias = True
    samples = 2

    random_data = torch.rand(samples, in_channels, 12, 24)
    target = torch.rand(samples, out_channels, 44, 44)

    pytorch_conv, custom_conv = _setup(True, in_channels, out_channels, kernel_size, stride, padding, dilation, bias,
                                       groups, output_padding)
    _test_forward(pytorch_conv, custom_conv, random_data)
    _test_backward(pytorch_conv, custom_conv, random_data, target, epsilon=1e-4, iterations=iterations)


def main():
    random_seed = 42
    random.seed(random_seed)
    torch.manual_seed(random_seed)

    test_conv2d()
    test_conv_transposed2d()


if __name__ == "__main__":
    main()
