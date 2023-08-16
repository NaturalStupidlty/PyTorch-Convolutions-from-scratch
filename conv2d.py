import math

import torch


class CustomConv2d(torch.nn.Module):
    """
    Custom implementation of 2D convolutional layer

    :param in_channels: number of input channels
    :param out_channels: number of output channels
    :param kernel_size: size of the kernel (filter)
    :param stride: stride of the kernel (filter)
    :param padding: padding of the kernel (filter)
    :param dilation: dilation of the kernel (filter)
    :param groups: number of blocked connections from input channels to output channels
    :param bias: whether to use bias or not

    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0,
                 dilation: int = 1, groups: int = 1, bias: bool = True):
        super(CustomConv2d, self).__init__()

        assert in_channels > 0, "in_channels must be greater than 0"
        assert out_channels > 0, "out_channels must be greater than 0"
        assert kernel_size > 0, "kernel_size must be greater than 0"
        assert stride > 0, "stride must be greater than 0"
        assert padding >= 0, "padding must be greater or equal to 0"
        assert dilation > 0, "dilation must be greater than 0"
        assert groups > 0, "groups must be greater than 0"

        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.cache = None

        self.weight = torch.nn.Parameter(torch.empty(out_channels, in_channels // groups, kernel_size, kernel_size))
        if bias:
            self.bias = torch.nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)

        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            torch.nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input_data: torch.Tensor):
        """
        Forward pass of the layer

        :param input_data: input data of shape (batch_size, in_channels, height, width)
        :return: output data of shape (batch_size, out_channels, out_height, out_width)
        """
        assert len(input_data.shape) == 4, "Input data must have shape (batch_size, in_channels, height, width)"

        batch_size, in_channels, in_height, in_width = input_data.shape
        out_channels, _, kernel_height, kernel_width = self.weight.shape

        out_height = ((in_height + 2 * self.padding - self.dilation * (kernel_height - 1) - 1) // self.stride) + 1
        out_width = ((in_width + 2 * self.padding - self.dilation * (kernel_width - 1) - 1) // self.stride) + 1

        padded_input = torch.nn.functional.pad(input_data, (self.padding, self.padding, self.padding, self.padding))
        self.cache = (input_data, padded_input, batch_size, out_channels, out_height, out_width)
        output = torch.zeros((batch_size, out_channels, out_height, out_width))

        for b in range(batch_size):
            for c_out in range(out_channels):
                for h_out in range(out_height):
                    for w_out in range(out_width):
                        output[b, c_out, h_out, w_out] = self._conv_forward(h_out, w_out, c_out, b, padded_input)

        return output

    def _conv_forward(self, h_out, w_out, c_out, b, padded_input):
        h_start = h_out * self.stride
        h_end = h_start + self.weight.size(2) * self.dilation
        w_start = w_out * self.stride
        w_end = w_start + self.weight.size(3) * self.dilation

        receptive_field = padded_input[b, :, h_start:h_end:self.dilation, w_start:w_end:self.dilation]
        weighted_receptive_field = receptive_field * self.weight[c_out]
        output = weighted_receptive_field.sum()

        if self.bias is not None:
            output += self.bias[c_out]

        return output

    def backward(self, grad_output: torch.Tensor):
        """
        Backward pass of the layer

        :param grad_output: gradient of the loss with respect to the output of the layer
        :return: gradient of the loss with respect to the input of the layer
        """

        assert len(grad_output.shape) == 4,\
            "Grad output must have shape (batch_size, out_channels, out_height, out_width)"

        input_data, padded_input, batch_size, out_channels, out_height, out_width = self.cache
        grad_input = torch.zeros_like(padded_input, device=self.weight.device)
        grad_weight = torch.zeros_like(self.weight)
        grad_bias = torch.zeros_like(self.bias) if self.bias is not None else None

        for b in range(batch_size):
            for c_out in range(out_channels):
                for h_out in range(out_height):
                    for w_out in range(out_width):
                        h_start = h_out * self.stride
                        h_end = h_start + self.weight.size(2) * self.dilation
                        w_start = w_out * self.stride
                        w_end = w_start + self.weight.size(3) * self.dilation

                        receptive_field = padded_input[b, :, h_start:h_end:self.dilation, w_start:w_end:self.dilation]

                        grad_input[b, :, h_start:h_end:self.dilation, w_start:w_end:self.dilation] += \
                            grad_output[b, c_out, h_out, w_out] * self.weight[c_out]

                        grad_weight[c_out] += grad_output[b, c_out, h_out, w_out] * receptive_field

                        if self.bias is not None:
                            grad_bias[c_out] += grad_output[b, c_out, h_out, w_out]

        self.weight.grad = grad_weight / batch_size
        if self.bias is not None:
            self.bias.grad = grad_bias / batch_size

        if self.padding > 0:
            grad_input = grad_input[:, :, self.padding:-self.padding, self.padding:-self.padding]

        assert (grad_input.shape == padded_input.shape),\
            f"Grad input shape ({grad_input.shape}) is not equal to input shape ({padded_input.shape})"

        return grad_input, grad_weight, grad_bias
