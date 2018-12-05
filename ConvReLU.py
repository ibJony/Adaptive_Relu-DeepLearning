import torch
import torch.nn as nn


# kernel_size must be an int 

# init values must be 0, 1, or 2
# 0: zero initialization
# 1: random initilaization 
# 2: xavier initialization 

# Make sure the paramters for initialization is the same as the conv2d 
# This ConvBlock is applied before conv2d
class ConvBlock(nn.Module):
  def __init__(self, in_channel, kernel_size, init=0, dilation=1, padding=0, stride=1):
    super(ConvBlock, self).__init__()
    self.kernel_size = kernel_size
    self.dilation = dilation
    self.padding = padding
    self.stride = stride
    self.thr = nn.Parameter(torch.zeros(in_channel, kernel_size * kernel_size))
    self.unf = nn.Unfold(kernel_size, dilation, padding, stride)
    if init == 1:
      self.thr.uniform_(-0.1, 0.1)
    else:
      if init == 2:
        torch.nn.init.xavier_uniform_(self.thr)

  def forward(self, input):
    # input has shape (N, C, size1, size2)
    # assert C == in_channel
    N = input.size(0)
    C = input.size(1)
    s1 = input.size(2)
    s2 = input.size(3)

    Fold_f = nn.Fold((s1, s2), self.kernel_size, self.dilation, self.padding, self.stride)

    unfold = self.unf(input)
    kernel_2 = self.kernel_size * self.kernel_size
    L = unfold.size(2)
    unfold = unfold.view(N, C, kernel_2, L)
    unfold = unfold.permute(0, 3, 1, 2) # size (N, L, C, kernel_2)

    after_max = torch.max(unfold, self.thr).permute(0, 2, 3, 1).contiguous().view(N, -1, L)

    fold_back = Fold_f(after_max)
    return fold_back
