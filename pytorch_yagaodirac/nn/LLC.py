import torch

class LLC(torch.nn.Module):
    '''Linear leaky clamp'''
    def __init__(self, size, * , a = -1., b = 2., negative_slope=0.01, epi = 1e-8):
        super(LLC, self).__init__()
        assert a<0
        assert b>=0.5
        assert negative_slope>0
        assert epi>0
        self.a = torch.nn.Parameter(torch.full((size, ), a, dtype = torch.float32), True)
        b_like = torch.sqrt(b-0.5)
        self.b_like = torch.nn.Parameter(torch.full((size, ), b_like, dtype = torch.float32), True)
        self.negative_slope = negative_slope
        self.epi = torch.abs(epi)
        pass
    def forward(self, x):
        with torch.no_grad():
            self.a = torch.abs(torch.a)
            self.a = self.a-self.epi
            self.a = torch.relu(self.a)
            self.a = self.a+self.epi
            self.a = -self.a # self.a < 0
            pass
        b = self.b_like*self.b_like+0.5      # b >= 0.5
        x = torch.abs(x)
        x = self.a*x+b
        # Then, double leaky relu to make the result almost inside 0 to 1
        x = torch.nn.functional.leaky_relu(x, negative_slope=self.negative_slope)
        x = 1 - torch.nn.functional.leaky_relu(1 - x, negative_slope=self.negative_slope)
        return x
        pass
    def __str__(self):
        return F'Linear leaky clamp. a: {self.a}  b: {self.b}  the boundary: {torch.abs(self.b/self.a)}'
    pass
