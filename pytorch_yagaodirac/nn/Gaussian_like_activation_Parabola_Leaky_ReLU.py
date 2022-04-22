import torch

class Gaussian_like_activation_Parabola_Leaky_ReLU(torch.nn.Module):
    def __init__(self, size, * , a = -1., negative_slope=0.01, epi = 1e-8):
        super(Gaussian_like_activation_Parabola_Leaky_ReLU, self).__init__()
        assert a < 0
        assert negative_slope > 0
        assert epi > 0
        sqrt_of_a = torch.sqrt(-a)
        self.sqrt_of_a = torch.nn.Parameter(torch.full((size, ), sqrt_of_a, dtype = torch.float32), True)
        self.negative_slope = negative_slope
        self.epi = epi
        pass
    def forward(self, x):
        a = self.sqrt_of_a * self.sqrt_of_a + self.epi
        one_minus_axx = -a*x*x + 1
        result = torch.nn.functional.leaky_relu(one_minus_axx, negative_slope=self.negative_slope)
        return result
        pass
    def __str__(self):
        a = self.sqrt_of_a*self.sqrt_of_a
        return F'Gaussian like activation. sqrt of a: {self.sqrt_of_a}  a: {a}  the boundary: {torch.abs(1/self.sqrt_of_a)}'
    pass
