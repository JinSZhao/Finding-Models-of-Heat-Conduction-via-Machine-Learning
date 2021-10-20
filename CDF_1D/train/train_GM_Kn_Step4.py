import scipy.io as sio
import numpy as np
import torch
import math
import torch.nn as nn
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler
import time
import pathlib
import os, glob
from collections import namedtuple

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2'
torch.set_default_dtype(torch.float64)


# ------------------------Model---------------------------------------

class CDFNet(nn.Module):
    def __init__(self, num1=50, num2=50):
        super(CDFNet, self).__init__()

        self.l0_G = torch.nn.Linear(1, num1)
        self.l1_G = torch.nn.Linear(num1, num1)
        self.l2_G = torch.nn.Linear(num1, num1)
        self.l3_G = torch.nn.Linear(num1, num1)
        self.l4_G = torch.nn.Linear(num1, 1)

        self.l0_M = torch.nn.Linear(2, num2)
        self.l1_M = torch.nn.Linear(num2, num2)
        self.l2_M = torch.nn.Linear(num2, num2)
        self.l3_M = torch.nn.Linear(num2, num2)
        self.l4_M = torch.nn.Linear(num2, 1)

    def G(self, q, Kn):
        z1_G = torch.tanh(self.l0_G(q))
        z2_G = torch.tanh(self.l1_G(z1_G))
        z3_G = torch.tanh(self.l2_G(z2_G))
        z4_G = torch.tanh(self.l3_G(z3_G))
        z5_G = F.softplus(self.l4_G(z4_G))
        z6_G = z5_G * Kn * Kn

        return z6_G

    def M(self, u, q, Kn):
        var_in = torch.cat((u, q), dim=1)
        z1_M = torch.tanh(self.l0_M(var_in))
        z2_M = torch.tanh(self.l1_M(z1_M))
        z3_M = torch.tanh(self.l2_M(z2_M))
        z4_M = F.softplus(self.l3_M(z3_M))
        z5_M = F.softplus(self.l4_M(z4_M))

        return z5_M

    def forward(self, u, q, lam, dt, K, n, Kn):
        u_p = torch.cat([u[:, 1:], u[:, 0:1]], dim=1)
        u_m = torch.cat([u[:, -1:], u[:, 0:-1]], dim=1)

        q_p = torch.cat([q[:, 1:], q[:, 0:1]], dim=1)
        q_m = torch.cat([q[:, -1:], q[:, 0:-1]], dim=1)

        Theta_inv_p = 1 / u_p
        Theta_inv_m = 1 / u_m

        G_c = self.G(q.reshape(K * n, 1), Kn)
        G_c = -1 * G_c.reshape(K, n)

        M_c = self.M(u.reshape(K * n, 1), q.reshape(K * n, 1), Kn)
        M_c = M_c.reshape(K, n)

        # Scheme
        q1 = (G_c * q - 0.5 * lam * (Theta_inv_p - Theta_inv_m)) / (G_c - dt * M_c)
        return q1


def read_data(load_fn):
    load_data = sio.loadmat(load_fn)
    u_0 = torch.from_numpy(load_data['u0']).float()
    q_0 = torch.from_numpy(load_data['q0']).float()

    q_1 = torch.from_numpy(load_data['q']).float()

    Kn = float(load_data['epsilon'][0][0])

    return u_0, q_0, q_1, Kn


def read_data_multi_step(load_fn, step=4):
    load_data = sio.loadmat(load_fn)
    num_init = load_data['num_init'][0, 0]
    QQ1 = load_data['q0']
    Q=np.array(QQ1,dtype=np.float64)
    UU1 = load_data['u0']
    U = np.array(UU1, dtype=np.float64)
    QQ2 = load_data['q']
    Q_plus=np.array(QQ2,dtype=np.float64)
    num_data, _ = Q.shape
    num_data_each_init = int(num_data / num_init)
    q0 = []
    q1 = []
    q2 = []
    u0 = []
    u1 = []

    q3 = []
    q4 = []
    q5 = []
    u2 = []
    u3 = []
    u4 = []

    for i in range(num_init):
        for j in range(num_data_each_init - step):
            q0.append(Q[i * num_data_each_init + j])
            q1.append(Q[i * num_data_each_init + j + 1])
            q2.append(Q[i * num_data_each_init + j + 2])
            q3.append(Q[i * num_data_each_init + j + 3])       # Addition
            q4.append(Q[i * num_data_each_init + j + 4])       # Addition
            q5.append(Q_plus[i * num_data_each_init + j + 4])  # Addition
            u0.append(U[i * num_data_each_init + j])
            u1.append(U[i * num_data_each_init + j + 1])
            u2.append(U[i * num_data_each_init + j + 2])       # Addition
            u3.append(U[i * num_data_each_init + j + 3])       # Addition
            u4.append(U[i * num_data_each_init + j + 4])       # Addition

    u_0 = torch.from_numpy(np.array(u0))
    q_0 = torch.from_numpy(np.array(q0))

    u_1 = torch.from_numpy(np.array(u1))
    q_1 = torch.from_numpy(np.array(q1))

    q_2 = torch.from_numpy(np.array(q2))

    u_2 = torch.from_numpy(np.array(u2))
    q_3 = torch.from_numpy(np.array(q3))

    u_3 = torch.from_numpy(np.array(u3))
    q_4 = torch.from_numpy(np.array(q4))

    u_4 = torch.from_numpy(np.array(u4))
    q_5 = torch.from_numpy(np.array(q5))

    Knload = load_data['epsilon'][0][0]
    Kn = np.array(Knload, dtype = np.float64)

    return u_0, q_0, u_1, q_1, q_2, u_2, q_3, u_3, q_4, u_4, q_5, Kn


def load_model(load_model_name):
    model = CDFNet()
    model.load_state_dict(torch.load(load_model_name))
    # model.eval()

    return model


def get_file_with_extension(file_path, extension):
    return [f for f in os.listdir(file_path) if f.endswith(extension)]


if __name__ == "__main__":

    # training group number
    train_group = list(range(11))
    print("train group numbers: ", train_group)

    # training Kn number
    train_Kn = [1e1]
    print("train Kn numbers: ", train_Kn[0])

    # current path
    curr_path = pathlib.Path(__file__).parent.absolute()

    # parent path
    parent_path = curr_path.parent

    # save net data path
    net_data_path = os.path.join(curr_path, "GMnetKn1e1_dataV3")
    if not os.path.exists(net_data_path):
        os.makedirs(net_data_path)

    # -------------Generate data-------
    pi = math.pi
    n = 160
    L = 2 * pi
    h = L / n
    x = np.linspace(-L / 2, L / 2 - h / 2, n)
    T = 1
    dt = 0.01
    lam = dt / h

    # DataStruct = namedtuple("matlab_data", "u_0, q_0, q_1, Kn")
    DataStruct = namedtuple("matlab_data", "u_0, q_0, u_1, q_1, q_2, u_2, q_3, u_3, q_4, u_4, q_5, Kn")

    training_data = []
    # loop over all the training group number
    data_group = [0, 5, 10]
    for data_group_number in data_group:
        matlab_data_path = os.path.join(parent_path, "data_all0428", "CFL{}".format(data_group_number))
        for train_group_number in train_group:

        # file name of training group number
        # file_path = os.path.join(matlab_data_path, str(train_group_number) + "st_init")
            file_path = os.path.join(matlab_data_path, "{}st_init".format(train_group_number))  # 文件路径 *st_init

        # loop over all the matlab data files
            matlab_data_file_names = get_file_with_extension(file_path, ".mat")
            for f in matlab_data_file_names:
            # u_0, q_0, q_1, Kn = read_data(os.path.join(file_path, f))
            # data = DataStruct(u_0, q_0, q_1, Kn)
                u_0, q_0, u_1, q_1, q_2, u_2, q_3, u_3, q_4, u_4, q_5, Kn = read_data_multi_step(os.path.join(file_path, f))
                data = DataStruct(u_0, q_0, u_1, q_1, q_2, u_2, q_3, u_3, q_4, u_4, q_5, Kn)

            # only add data with training Kn number into our training data
                if data.Kn in train_Kn:
                    training_data.append(data)

    u_0 = training_data[0].u_0
    q_0 = training_data[0].q_0
    u_1 = training_data[0].u_1
    q_1 = training_data[0].q_1
    q_2 = training_data[0].q_2
    u_2 = training_data[0].u_2
    q_3 = training_data[0].q_3
    u_3 = training_data[0].u_3
    q_4 = training_data[0].q_4
    u_4 = training_data[0].u_4
    q_5 = training_data[0].q_5
    Kn = train_Kn[0]
    for i in range(len(training_data) - 1):
        u_0 = torch.cat((u_0, training_data[i + 1].u_0))
        q_0 = torch.cat((q_0, training_data[i + 1].q_0))
        u_1 = torch.cat((u_1, training_data[i + 1].u_1))
        q_1 = torch.cat((q_1, training_data[i + 1].q_1))
        q_2 = torch.cat((q_2, training_data[i + 1].q_2))
        u_2 = torch.cat((u_2, training_data[i + 1].u_2))
        q_3 = torch.cat((q_3, training_data[i + 1].q_3))
        u_3 = torch.cat((u_3, training_data[i + 1].u_3))
        q_4 = torch.cat((q_4, training_data[i + 1].q_4))
        u_4 = torch.cat((u_4, training_data[i + 1].u_4))
        q_5 = torch.cat((q_5, training_data[i + 1].q_5))

    # ----------------------Train---------------------------------

    # model = CDFNet()
    # # use pre-trained model
    curr_path = pathlib.Path(__file__).parent.absolute()
    net_path = os.path.join(curr_path, "GMnetKn1e1_dataV3")
    model = load_model(os.path.join(net_path, "net_params.pkl"))
    model.train()

    total_size = u_0.size()[0]
    batch_size = int(total_size / 480)
    L = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)

    num_iter = 500
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_iter, eta_min=0)
    loss_history = np.zeros(num_iter)

    start = time.time()
    for epoch in range(num_iter):

        permutation = torch.randperm(total_size)

        for i in range(0, total_size, batch_size):

            indices = permutation[i: i + batch_size]

            batch_u_0, batch_q_0, batch_q_1 = u_0[indices], q_0[indices], q_1[indices]
            batch_u_1, batch_q_2 = u_1[indices], q_2[indices]
            batch_u_2, batch_q_3 = u_2[indices], q_3[indices]
            batch_u_3, batch_q_4 = u_3[indices], q_4[indices]
            batch_u_4, batch_q_5 = u_4[indices], q_5[indices]

            batch_qpred = model(batch_u_0, batch_q_0, lam, dt, batch_size, n, Kn)

            batch_qpred_2 = model(batch_u_1, batch_qpred, lam, dt, batch_size, n, Kn)

            batch_qpred_3 = model(batch_u_2, batch_qpred_2, lam, dt, batch_size, n, Kn)

            batch_qpred_4 = model(batch_u_3, batch_qpred_3, lam, dt, batch_size, n, Kn)

            batch_qpred_5 = model(batch_u_4, batch_qpred_4, lam, dt, batch_size, n, Kn)

            # # first stage
            # loss = L(batch_qpred, batch_q_1)
            # # second stage
            # loss = L(batch_qpred, batch_q_1) + L(batch_qpred_2, batch_q_2)
            if epoch < 2:
                loss = L(batch_qpred, batch_q_1)
            # elif epoch < 30:
            #     loss = L(batch_qpred, batch_q_1) + L(batch_qpred_2, batch_q_2)
            # elif epoch < 40:
            #     loss = L(batch_qpred, batch_q_1) + L(batch_qpred_2, batch_q_2) + L(batch_qpred_3, batch_q_3)
            # elif epoch < 50:
            #     loss = L(batch_qpred, batch_q_1) + L(batch_qpred_2, batch_q_2) + L(batch_qpred_3, batch_q_3) + L(batch_qpred_4, batch_q_4)
            else:
                loss = L(batch_qpred, batch_q_1) + L(batch_qpred_2, batch_q_2) + L(batch_qpred_3, batch_q_3) + L(
                    batch_qpred_4, batch_q_4) + L(batch_qpred_5, batch_q_5)


            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        scheduler.step()

        loss_history[epoch] = loss.item()
        if epoch % 8 == 0:
            end = time.time()
            for param_group in optimizer.param_groups:
                lr = param_group['lr']
            resultsG1 = model.G(torch.tensor([[0.08]]), Kn)
            resultsG2 = model.G(torch.tensor([[0.07]]), Kn)
            resultsM = model.M(torch.tensor([[0.40]]), torch.tensor([[0.07]]), Kn)
            print(-1*resultsG1)
            print(-1 * resultsG2)
            print(resultsM)
            print('epoch: {}, loss: {:.2E}, lr: {:.2E}, elapsed time: {:.2f}'.format(epoch, loss_history[epoch], lr,
                                                                                     end - start))

        # save neural network
        #     torch.save(model.state_dict(), os.path.join(net_data_path, "net_params_epoch" + str(epoch) + ".pkl"))

        if loss<1e-15:
            break

    # save neural network
    torch.save(model.state_dict(), os.path.join(net_data_path, "net_params.pkl"))









