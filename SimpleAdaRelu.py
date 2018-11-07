import torch
import torch.nn as nn



# This just set a learnable relu threshold without sharing 
class AdaptiveRelu(nn.Module):
	def __init__(self):
		super(AdaptiveRelu, self).__init__()
		self.thr = nn.Parameter(torch.tensor(0.0))
	def forward(self, x):
		x = nn.ReLU()(x - self.thr) + self.thr
# 		print(self.thr)
		return x
