import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from scipy.integrate import quad
import pathlib
import os
import sys

sys.path.append("..")
from train import train_GM_Kn_Step4
import torch.optim.lr_scheduler as lr_scheduler

torch.set_default_dtype(torch.float64)

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2'
torch.set_default_dtype(torch.float64)


# Learn a convex function S(w) based on its derivative S_ww

# ------------------------------------------------------------
# In CDF model, we only use S_w and it is unique up to a constant A
# We learn S_w as follows:
# (1) learn a convex function S(w)
# (2) compute S_w(w)-S_w(0)


class fNet(nn.Module):

    def __init__(self, num=30, dim_input=1, dim_output=1):
        super(fNet, self).__init__()
        self.dim_input = dim_input
        self.dim_output = dim_output
        # self.bn_layer_f = torch.nn.BatchNorm1d(num, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        self.l1 = nn.Sequential(
            nn.Linear(dim_input, num),
            nn.Tanh(),
            nn.Linear(num, num),
            nn.Tanh(),
            nn.Linear(num, num),
            nn.Tanh(),
            nn.Linear(num, dim_output)
        )

    def forward(self, x):
        # return self.l2(x)
        return self.l1(x)


def load_fmodel(load_model_name):
    fmodel = fNet()
    fmodel.load_state_dict(torch.load(load_model_name))
    # fmodel.eval()
    return fmodel


curr_path = pathlib.Path(__file__).parent.absolute()
net_path = os.path.join(curr_path, "GMnetKn1e1_dataV3")

Model = train_GM_Kn_Step4.load_model(os.path.join(net_path, "net_params.pkl"))


def f_prime_exact(q, Kn):
    return (-1) * Model.G(q, Kn)


# --------------------------------
# Given derivative of f (i.e. f'), learn f (represented by fNet)
# --------------------------------
if __name__ == "__main__":

    Kn = 1e1

    m = 0.18
    n = 1200
    q = np.linspace(-m, m, n)

    # q_tensor = torch.tensor(q, requires_grad=True).float().reshape(n, 1)
    q_tensor = torch.tensor(q, requires_grad=True, dtype=torch.float64).reshape(n, 1)
    # q_tensor = torch.tensor(q, requires_grad=True).double().reshape(n, 1)
    q_zero_tensor = torch.zeros(q_tensor.shape, requires_grad=True)

    # fNet = fNet()
    # # use pre-trained model
    # curr_path = pathlib.Path(__file__).parent.absolute()
    net_path_load = os.path.join(curr_path, "GMnetKn1e1_dataV3")
    fNet = load_fmodel(os.path.join(net_path_load, "fnet_params.pkl"))


    L = nn.MSELoss()
    optimizer = torch.optim.SGD(fNet.parameters(), lr=5e-7, momentum=0.9)
    # Kn=1e-2, choose lr=8e-8 and
    # Kn=1e-1, choose lr=5e-4
    # Kn=1e0,  choose lr=1e-4 and Adam
    # Kn=1e1, choose lr=5e-7
    # Kn=1e2, choose lr=1e-4 and Adam
    #
    # optimizer = torch.optim.Adam(fNet.parameters(), lr=1e-4) #2.70e-7# This is 1e2!!!!

    num_iter = 100000
    total_size = n
    batch_size = int(total_size/1)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_iter, eta_min=0)
    loss_history = np.zeros(num_iter)

    m_q = 0.18
    n_q = 1200
    q_test = np.linspace(-m_q, m_q, n_q)
    # q_test_tensor = torch.tensor(q_test).float().reshape(n_q, 1)
    q_test_tensor = torch.tensor(q_test).double().reshape(n_q, 1)
    q_test_zero = torch.zeros(q_test_tensor.shape)

    for epoch in range(num_iter):

        derivative = \
        torch.autograd.grad(fNet(q_tensor), q_tensor, grad_outputs=torch.ones(q_tensor.shape), create_graph=True)[0]
        loss = L(f_prime_exact(q_tensor, Kn), derivative)

        loss_history[epoch] = loss.item()

        if epoch % 1000 == 0:  #
            lr = optimizer.param_groups[0]['lr']
            print('epoch: {}, loss: {:.2E}, lr: {:.2E}'.format(epoch, loss_history[epoch], lr))

        if epoch % (num_iter / 5) == 0:
            plt.figure()
            qd = (fNet(q_test_tensor) - fNet(q_test_zero)).detach().numpy()
            plt.plot(q_test, qd, 'r+', markersize=2)
            plt.show()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        scheduler.step()


        if loss<1e-8:
            break

    # save neural network
    torch.save(fNet.state_dict(), os.path.join(net_path, "fnet_params.pkl"))

    plt.figure()
    qd = (fNet(q_test_tensor) - fNet(q_test_zero)).detach().numpy()
    plt.plot(q_test, qd, 'o', markersize=2)
    plt.show()







