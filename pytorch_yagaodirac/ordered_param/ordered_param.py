import torch

class ordered_param(torch.nn.Module):
    def __init__(self, real_val:torch.Tensor, order = 3):
        '''
        example:
        w = ordered_param(...)
        x = w() * x + b# Extra parenthesis needed.
             ^^ Notice the extra parenthesis comparing to the normal pytorch convention.
        real_val is torch.Tensor or parameter.
        Notice: I recommend the order be greater than 1, or less than 0.
        Generally order = 3 would speed up the training by 2.5x
        order = 5 would speed up the training by 3.5x
        order = -1 would speed up similar to order = 2.3
        order = -3 would speed up similar to order = 4.5
        All the data are from a rough test with only 1 layer and 1 netron.
        If the abs of order is too big, the IEEE.754 floating point number may crash, while you don't obtain too drastic performence promotion
        See the "higher order of weight test.ipynb" file in this folder.
        '''

        assert order <0. or order >1., "0 may crash the algorithm. 0<order<=1 slows down the training. For more details, please run the .ipynb file in this folder."
        super(ordered_param, self).__init__()
        self.order = torch.nn.Parameter(torch.tensor([order], dtype = torch.float32), requires_grad=False)
        self.data = torch.nn.Parameter(torch.pow(real_val, 1/order))
        pass
    def forward(self):
        return torch.pow(self.data, self.order)
        pass
    pass#class

if 0:
    p1 = ordered_param(torch.tensor([8])).cuda()
    r1 = p1()

    fdsjkfld = 545345
    pass






class third_order_param(torch.nn.Module):
    def __init__(self, real_val:torch.Tensor):
        '''
        example:
        w = ordered_param(...)
        x = w() * x + b# Extra parenthesis needed.
             ^^ Notice the extra parenthesis comparing to the normal pytorch convention.
        real_val is torch.Tensor or parameter.
        Notice: I recommend the order be greater than 1, or less than 0.
        Generally order = 3 would speed up the training by 2.5x
        order = 5 would speed up the training by 3.5x
        order = -1 would speed up similar to order = 2.3
        order = -3 would speed up similar to order = 4.5
        All the data are from a rough test with only 1 layer and 1 netron.
        If the abs of order is too big, the IEEE.754 floating point number may crash, while you don't obtain too drastic performence promotion
        See the "higher order of weight test.ipynb" file in this folder.
        '''
        super(third_order_param, self).__init__()
        self.data = torch.nn.Parameter(torch.pow(real_val, 1/3))
        pass
    def forward(self):
        return self.data*self.data*self.data
        pass
    def get_order(self):
        return 3
    def __str__(self):
        return F'Third order parameter. Real value is {self()}'
    pass#class

if 0:
    p1 = third_order_param(torch.tensor([8])).cuda()
    r1 = p1()

    fdsjkfld = 545345
    pass






class fifth_order_param(torch.nn.Module):
    def __init__(self, real_val:torch.Tensor):
        '''
        example:
        w = ordered_param(...)
        x = w() * x + b# Extra parenthesis needed.
             ^^ Notice the extra parenthesis comparing to the normal pytorch convention.
        real_val is torch.Tensor or parameter.
        Notice: I recommend the order be greater than 1, or less than 0.
        Generally order = 3 would speed up the training by 2.5x
        order = 5 would speed up the training by 3.5x
        order = -1 would speed up similar to order = 2.3
        order = -3 would speed up similar to order = 4.5
        All the data are from a rough test with only 1 layer and 1 netron.
        If the abs of order is too big, the IEEE.754 floating point number may crash, while you don't obtain too drastic performence promotion
        See the "higher order of weight test.ipynb" file in this folder.
        '''
        super(fifth_order_param, self).__init__()
        self.data = torch.nn.Parameter(torch.pow(real_val, 1/5))
        pass
    def forward(self):
        return self.data*self.data*self.data*self.data*self.data
        pass
    def get_order(self):
        return 5
    def __str__(self):
        return F'Fifth order parameter. Real value is {self()}'
    pass#class

if 0:
    p1 = fifth_order_param(torch.tensor([243])).cuda()
    r1 = p1()

    fdsjkfld = 545345
    pass

















