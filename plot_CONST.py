import torch

'''
Activations
'''


class ReLUActivation(torch.autograd.Function):

    def __str__(self):
        return 'relu'

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input.clamp(min=0)

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input < 0] = 0
        return grad_input


def rbf(x): 
    return torch.exp(- x.pow(2))

relu = ReLUActivation.apply

def tanh(x): 
    return x.tanh(x)

def softrelu(x): 
    return torch.log(1.0 + x.exp())


'''
Data
'''

# f

def f(x):
    return 2 * torch.exp(- x.pow(2)) * torch.sin(5 * x)

X_f = torch.tensor([-2, -1.8, -1.5, -1, -0.8, -0.05,
                    0.05, 1.1, 1.8, 2.1]).unsqueeze(1)
Y_f = f(X_f)

X_plot_f = torch.linspace(-5, 5, steps=1000).unsqueeze(1)
Y_plot_f = f(X_plot_f)

X_id_f = torch.tensor([-2.1, -1.6, -0.8, -0.3, 0.5, 1.6, 1.8]).unsqueeze(1)
Y_id_f = f(X_id_f)

X_ood_f = torch.tensor([-4, -3.1, -2.4, 2.7, 3, 4.4]).unsqueeze(1)
Y_ood_f = f(X_ood_f)

# g

def g(x):
    return - 0.6666 * x.pow(4) + 4/3 * x.pow(2) + 1


X_g = torch.tensor([-2, -1.8, -1.1, 1.1, 1.8, 2]).unsqueeze(1)
Y_g = g(X_g)

X_plot_g = torch.linspace(-5, 5, steps=1000).unsqueeze(1)
Y_plot_g = g(X_plot_g)

X_id_g = torch.tensor([-1.9, -1.5, 0.5, 0.0, 0.5, 1.5, 1.9]).unsqueeze(1)
Y_id_g = g(X_id_g)

X_ood_g = torch.tensor([-4, -3, -2.5, 2.5, 3, 4]).unsqueeze(1)
Y_ood_g = g(X_ood_g)
