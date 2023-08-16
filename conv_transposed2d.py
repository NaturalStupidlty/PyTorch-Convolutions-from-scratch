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
                    for w_out in range(out_width):
                        h_start_out = h_out * self.stride
                        w_start_out = w_out * self.stride

                        # Calculate input slice using dilation
                        h_in = h_start_out - self.dilation * (kernel_height - 1)
                        w_in = w_start_out - self.dilation * (kernel_width - 1)

                        # Initialize output value for the current position
                        out_value = 0.0

                        # Iterate over the kernel and perform the convolution
                        for c_in in range(in_channels):
                            for k_row in range(kernel_height):
                                for k_col in range(kernel_width):
                                    h_in_current = h_in + k_row * self.dilation
                                    w_in_current = w_in + k_col * self.dilation

                                    # Check if the current indices are within valid range
                                    if 0 <= h_in_current < in_height and 0 <= w_in_current < in_width:
                                        out_value += padded_input[b, c_in, h_in_current, w_in_current] * self.weight[
                                            c_in, c_out, k_row, k_col]

                        output[b, c_out, h_out, w_out] = out_value
        output /= batch_size

        if self.bias is not None:
            output += self.bias.view(1, -1, 1, 1)  # Add bias

        return output

    def backward(self, grad_output: torch.Tensor):
        """
        Backward pass of the layer

        :param grad_output: gradient of the loss with respect to the output of the layer
        :return: gradient of the loss with respect to the input of the layer
        """

        assert len(grad_output.shape) == 4, "Grad output must have shape (batch_size, out_channels, out_height, out_width)"

        input_data, batch_size, out_channels, out_height, out_width = self.cache
        grad_input = torch.zeros_like(input_data, device=self.weight.device)
        grad_weight = torch.zeros_like(self.weight)
        grad_bias = torch.zeros_like(self.bias) if self.bias is not None else None

        kernel_height = self.weight.size(2)
        kernel_width = self.weight.size(3)
        in_channels = self.weight.size(0)
        in_height = input_data.size(2)
        in_width = input_data.size(3)

        for b in range(batch_size):
            for c_out in range(out_channels):
                for h_out in range(out_height):
                    for w_out in range(out_width):
                        h_start = h_out * self.stride
                        w_start = w_out * self.stride
                        for k_row in range(kernel_height):
                            for k_col in range(kernel_width):
                                h_in = h_start + k_row - self.padding
                                w_in = w_start + k_col - self.padding
                                for c_in in range(in_channels):  # Loop over input channels
                                    if 0 <= h_in < in_height and 0 <= w_in < in_width:
                                        grad_input[b, c_in, h_in, w_in] += \
                                            self.weight[c_in, c_out, k_row, k_col] * grad_output[b, c_out, h_out, w_out]
                                        grad_weight[c_in, c_out, k_row, k_col] += \
                                            input_data[b, c_in, h_in, w_in] * grad_output[b, c_out, h_out, w_out]

                        if self.bias is not None:
                            grad_bias[c_out] += grad_output[b, c_out, h_out, w_out]

        self.weight.grad = grad_weight / batch_size
        if self.bias is not None:
            self.bias.grad = grad_bias / batch_size

        return grad_input, grad_weight, grad_bias
