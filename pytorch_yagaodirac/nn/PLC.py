import torch

class PLC(torch.nn.Module):
    '''
    PLC == Paradola leaky clamp
    The Old name for this was PDLR, which means Parabola double leady relu
    '''
    def __init__(self, size, * , a = -1., c = 2., negative_slope=0.01, epi = 1e-8):
        super(PLC, self).__init__()
        assert a < 0
        assert negative_slope > 0
        assert epi > 0
        assert c > epi
        sqrt_of_a = torch.sqrt(-a)
        sqrt_of_c = torch.sqrt(c)
        self.sqrt_of_a = torch.nn.Parameter(torch.full((size,), sqrt_of_a, dtype=torch.float32), True)
        self.c = torch.nn.Parameter(torch.full((size, ), sqrt_of_c, dtype = torch.float32), True)
        self.negative_slope = negative_slope
        self.epi = epi
        pass
    def forward(self, x):
        a = self.sqrt_of_a * self.sqrt_of_a + self.epi
        c = self.sqrt_of_c * self.sqrt_of_c + self.epi
        result = -a*x*x + c
        # Then, double relu to make the result almost inside 0 to 1
        result = torch.nn.functional.leaky_relu(result, negative_slope=self.negative_slope)
        result = 1 - torch.nn.functional.leaky_relu(1 - result, negative_slope=self.negative_slope)
        return result
        pass
    def __str__(self):
        a = self.sqrt_of_a*self.sqrt_of_a
        return F'Parabola double leady relu. sqrt of a: {self.sqrt_of_a}  a: {a}  c: {self.c}  the boundary: {torch.abs(torch.sqrt(self.c)/self.sqrt_of_a)}'
    pass
