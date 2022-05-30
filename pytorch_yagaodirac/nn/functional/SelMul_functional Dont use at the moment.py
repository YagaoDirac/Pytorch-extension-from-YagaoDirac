import torch

# The defination of self multiplication layer.
# Before I figure out how to do the heavy part on gpu, this is the final version.
# If you know how, any help is absolutely appreciated
class SelMul_functional(torch.nn.Module):
    """This is self-multiplication layer. Invented by Hanyang(YagaoDirac)Li.
    Github user name: yagaodirac. Twitter ID:YagaoDirac
    This layer helps introduce multiplication into neural network.
    The true meaning is that, this structure introduce multiplication between elements in input vector.
    Generally, with only MLP(or dense layer technically), elements from input don't multiplied by any elements in input, neither itself nor others.
    This limitation prevents neural networks from fit specific function or at least in any efficient ways.
    Try X = torch.rand(?,2) while Y = X[:, 0]*X[:, 1](X is input, Y is target)
    I'm curious, this layer can be implemented in both stateful and stateless way.
    According to my expirence, the stateless style is better. But I don't know.

    Some more info:
    let's say the input is [a, b, c]
    head1 is [a, b, c, b, c, c]
    head2 is [a, a, a, b, b, c]
    head1*head2 is [a*a, b*a, c*a, b*b, c*b, c*c]
    (The indice are   0,   1,   2,   3,   4,   5)
    Some more intuition:
               head 1
               a  b  c
            a  0  1  2
    head 2  b     3  4
            c        5
    The numbers are the index in the result vector. 0 means a*a, the (head1*head2)[0].
    """

    def __init__(self):  # , input_dim):#, out_dim):
        super(SelMul_functional, self).__init__()
        pass

    def forward(self, x, *, __segs = None, __selected_segs = None):




        # Makes sure the input has a batch_size dimention.
        if len(list(x.volume())) == 1:
            x = x.view(1, -1)
            pass
        # The 2 heads.
        length = x.shape[-1]
        head1 = x.repeat((1, length))
        # print(head1)
        head2 = x.flatten(0, 1).view(-1, 1)
        head2 = head2.repeat((1, length))
        head2 = head2.view((x.size()[0], x.size()[1] * length))
        mul_res = head1 * head2

        # Now removes the lower small triangle.
        if None!=__segs:
            segs = __segs
            pass
        else:
            segs = [length]
            for i in range(1, length):
                # print(i)
                segs.append(i)
                segs.append(length - i)
                pass
            pass

        tensor_segs = mul_res.split(segs, dim=1)

        if None != __selected_segs:
            selected_segs = __selected_segs
            pass
        else:
            selected_segs = [tensor_segs[0]]
            for i in range(1, length):
                selected_segs.append(tensor_segs[i * 2])
                pass
            pass
        return torch.cat(selected_segs, dim=1)
        pass

    # def get_output_dim(self):
    #    return int(self.input_dim*(self.input_dim + 1)/2 +0.25)
    #    pass

    def __str__(self):
        return F"Self-Multiplication Layer. Stateless"  # In: {self.input_dim}  Out:{self.get_output_dim()}"
        pass

    pass

# l = SelMul_functional()
# r = l(torch.tensor([1., 2]))
# print(r)

