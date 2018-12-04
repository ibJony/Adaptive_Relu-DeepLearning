class AdaptiveRelu(nn.Module):
    def __init__(self, in_channel, random):
        super(AdaptiveRelu, self).__init__()
        self.in_channel = in_channel
        if random:
        	self.thr = nn.Parameter(torch.zeros(self.in_channel).uniform_(-0.1,0.1))
        else:
        	self.thr = nn.Parameter(torch.zeros(self.in_channel))
        

    def forward(self, x):
        # x has the shape (N, C, s1, s2)
        # assert self.inchannel == C
        N = x.size(0)
        C = x.size(1)
        # print(C)
        # print(self.in_channel)
        s1 = x.size(2)
        s2 = x.size(3)
        after_permute = x.permute(1,0,2,3).contiguous().view(C,-1)
        pm = torch.max(after_permute, self.thr.view(C, -1))

        pm = pm.view(C, N, s1, s2).permute(1,0,2,3)
        return pm
