
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.distributions as ds
from torch.autograd import Variable, grad

from torch.autograd.gradcheck import zero_gradients



# x = Variable(torch.ones(3), requires_grad=True)
# y = x.pow(3).sum() + x[0] * x[1]

# g = grad(y, x, create_graph=True)[0]
# print(g) # g = 3

# g2 = grad(g[2], x)
# print(g2) # g2 = 6


# exit(0)

# x = torch.ones(2, 2, requires_grad=True)

mu_dim = 5
mu = torch.arange(1, mu_dim + 1, requires_grad=True, dtype=torch.float)

logp = mu.pow(2).sum().log()

dlogp = grad(logp, mu, create_graph=True)[0]

# compute the second order derivative w.r.t. each parameter
diag_hess = []

for j in range(mu_dim):
    drv = grad(dlogp[j], mu, create_graph=True)
    diag_hess.append(drv)
    # print(drv)
    # print()


#  Computes trace of hessian of f w.r.t. x (x is 1D)
def trace_hessian(f, x):
    dim = x.shape[0]
    df = grad(f, x, create_graph=True)[0]
    tr_hess = 0
    # iterate over every entry in df/dx and compute derivate
    for j in range(dim):
        d2fj = grad(df[j], x, create_graph=True)[0]
        # add d2f/dx2 to trace
        tr_hess += d2fj[j]
        diag_hess.append(drv)
    return tr_hess.detach()

tr = trace_hessian(logp, mu)
print(tr)


# for grd, param in zip(dlogp, mu):
#     print(grd, param)
#     drv = grad(grd, param, create_graph=True)
#     diag_hess.append(drv)

    
# diag_hess = torch.tensor(diag_hess, dtype=torch.float)
# print(diag_hess)


   




def diag_hessian():
    pass



# f_1 = grad(y, x)
# f_2 = grad(f_1, x)


# print(f_1)
# print(f_2)


# def compute_jacobian(inputs, output):
# 	"""
# 	:param inputs: Batch X Size (e.g. Depth X Width X Height)
# 	:param output: Batch X Classes
# 	:return: jacobian: Batch X Classes X Size
# 	"""
# 	assert inputs.requires_grad

# 	num_classes = output.size()[1]

# 	jacobian = torch.zeros(num_classes, *inputs.size())
# 	grad_output = torch.zeros(*output.size())
# 	if inputs.is_cuda:
# 		grad_output = grad_output.cuda()
# 		jacobian = jacobian.cuda()

# 	for i in range(num_classes):
# 		zero_gradients(inputs)
# 		grad_output.zero_()
# 		grad_output[:, i] = 1
# 		output.backward(grad_output, retain_variables=True)
# 		jacobian[i] = inputs.grad.data

# 	return torch.transpose(jacobian, dim0=0, dim1=1)

