import math
import torch


class CustomConvTransposed2d(torch.nn.Module):
    """
    Custom implementation of 2D transposed convolutional layer

    :param in_channels: number of input channels
    :param out_channels: number of output channels
    :param kernel_size: size of the kernel (filter)
    :param stride: stride of the kernel (filter)
    :param padding: padding of the kernel (filter)
    :param output_padding: additional size added to the output shape
    :param dilation: dilation of the kernel (filter)
    :param groups: number of blocked connections from input channels to output channels
    :param bias: whether to use bias or not
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1,
                 padding: int = 0, output_padding: int = 0, dilation: int = 1, groups: int = 1, bias: bool = True):
        super(CustomConvTransposed2d, self).__init__()

        assert in_channels > 0, "in_channels must be greater than 0"
        assert out_channels > 0, "out_channels must be greater than 0"
        assert kernel_size > 0, "kernel_size must be greater than 0"
        assert stride > 0, "stride must be greater than 0"
        assert padding >= 0, "padding must be greater or equal to 0"
        assert output_padding >= 0, "output_padding must be greater or equal to 0"
        assert dilation > 0, "dilation must be greater than 0"
        assert groups > 0, "groups must be greater than 0"

        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.dilation = dilation
        self.groups = groups
        self.cache = None

        self.weight = torch.nn.Parameter(torch.Tensor(in_channels, out_channels // groups, kernel_size, kernel_size))
        if bias:
            self.bias = torch.nn.Parameter(torch.Tensor(out_channels))
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
        in_channels, out_channels, kernel_height, kernel_width = self.weight.shape

        out_height = (in_height - 1) * self.stride - 2 * self.padding + self.dilation * (
                kernel_height - 1) + self.output_padding + 1
        out_width = (in_width - 1) * self.stride - 2 * self.padding + self.dilation * (
                kernel_width - 1) + self.output_padding + 1

        output = torch.zeros((batch_size, out_channels, out_height, out_width))
        self.cache = (input_data, batch_size, out_channels, out_height, out_width)
        padded_input = torch.nn.functional.pad(input_data, (self.padding, self.padding, self.padding, self.padding))

        for b in range(batch_size):
            for c_out in range(out_channels):
                for h_out in range(out_height):
                    h_start_out = h_out * self.stride
                    for w_out in range(out_width):
                        w_start_out = w_out * self.stride
                        output[b, c_out, h_out, w_out] = self._forward_conv(h_start_out, w_start_out, c_out, b,
                                                                            padded_input, in_channels, in_height,
                                                                            in_width, kernel_height, kernel_width)
        output /= batch_size

        if self.bias is not None:
            output += self.bias.view(1, -1, 1, 1)

        return output

    def _forward_conv(self, h_start_out, w_start_out, c_out, b, padded_input, in_channels, in_height, in_width,
                      kernel_height, kernel_width):
        out_value = 0.0
        h_in = h_start_out - self.dilation * (kernel_height - 1)
        w_in = w_start_out - self.dilation * (kernel_width - 1)
        for c_in in range(in_channels):
            for k_row in range(kernel_height):
                for k_col in range(kernel_width):
                    h_in_current = h_in + k_row * self.dilation
                    w_in_current = w_in + k_col * self.dilation
                    if 0 <= h_in_current < in_height and 0 <= w_in_current < in_width:
                        out_value += padded_input[b, c_in, h_in_current, w_in_current] * self.weight[
                            c_in, c_out, k_row, k_col]

        return out_value

    def backward(self, grad_output: torch.Tensor):
        """
        Backward pass of the layer

        :param grad_output: gradient of the loss with respect to the output of the layer
        :return: gradient of the loss with respect to the input of the layer
        """

        assert len(grad_output.shape) == 4,\
            "Grad output must have shape (batch_size, out_channels, out_height, out_width)"

        input_data, batch_size, out_channels, out_height, out_width = self.cache
        grad_bias = torch.zeros_like(self.bias) if self.bias is not None else None
        grad_input = torch.zeros_like(input_data, device=self.weight.device)
        grad_weight = torch.zeros_like(self.weight)

        for b in range(batch_size):
            for c_out in range(out_channels):
                for h_out in range(out_height):
                    h_start = h_out * self.stride
                    for w_out in range(out_width):
                        w_start = w_out * self.stride
                        grad_input, grad_weight, grad_bias = self._conv_backward(grad_output, input_data, grad_input,
                                                                                 grad_bias, grad_weight, h_start,
                                                                                 w_start, h_out, w_out, c_out, b)
                        if self.bias is not None:
                            grad_bias[c_out] += grad_output[b, c_out, h_out, w_out]

        self.weight.grad = grad_weight / batch_size
        if self.bias is not None:
            self.bias.grad = grad_bias / batch_size

        return grad_input, grad_weight, grad_bias

    def _conv_backward(self, grad_output: torch.Tensor, input_data: torch.Tensor, grad_input: torch.Tensor,
                       grad_bias: torch.Tensor, grad_weight: torch.Tensor, h_start: int, w_start: int, h_out: int,
                       w_out: int, c_out: int, b: int):

        for k_row in range(self.weight.size(2)):
            h_in = h_start + k_row - self.padding
            for k_col in range(self.weight.size(3)):
                w_in = w_start + k_col - self.padding
                for c_in in range(self.weight.size(0)):
                    if 0 <= h_in < input_data.size(2) and 0 <= w_in < input_data.size(3):
                        grad_input[b, c_in, h_in, w_in] += \
                            self.weight[c_in, c_out, k_row, k_col] * grad_output[b, c_out, h_out, w_out]
                        grad_weight[c_in, c_out, k_row, k_col] += \
                            input_data[b, c_in, h_in, w_in] * grad_output[b, c_out, h_out, w_out]

        return grad_input, grad_weight, grad_bias

