import scipy.io as sio
import numpy as np
import torch
import matplotlib.pyplot as plt
import math
import pathlib
import os
import sys

sys.path.append("..")
from train import train_GM_Kn_Step4
from train import trainf_Kn

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2'
torch.set_default_dtype(torch.float64)
# Basic parameters
# ----------------------------
# we take the results from
#Kn1e-2
#Kn1e-1
#Kn1e0V3
#Kn1e1V3
#Kn1e2V2
#----------------------------
pi = math.pi

# Kn number
Kn = 1e1
initial_Kn_name = "st_init_Kn1e1.mat"

# predicition group number
predicition_group = [10]
print("predicition group numbers: ", predicition_group)

fNet_file = "fnet_params.pkl"
CDFNet_file = "net_params.pkl"


def read_data(load_fn, T_num, i_num):
    load_data = sio.loadmat(load_fn)

    number = (i_num - 1) * 100 + T_num * 10 - 1 #if T_num > 1 else 0
    u_1 = torch.from_numpy(load_data['u'])[number].float()
    q_1 = torch.from_numpy(load_data['q'])[number].float()

    return u_1, q_1

def read_data_inintial(load_fn):
    load_data = sio.loadmat(load_fn)

    u_0 = torch.from_numpy(load_data['u0'])[0].double()
    q_0 = torch.from_numpy(load_data['q0'])[0].double()

    return u_0, q_0


def read_v_data(load_fn):
    load_data = sio.loadmat(load_fn)
    v = torch.from_numpy(load_data['Leg']).float()

    return v

# curr_path = .../public/predict
curr_path = pathlib.Path(__file__).parent.absolute()

# parent_path = .../public
parent_path = curr_path.parent

# matlab_path = .../public/old_matlab
matlab_path_data = os.path.join(parent_path, "data_all0428T2", "CFL7")

predict_path = os.path.join(parent_path, "predict")

# net_data_path = .../public/train/net_data
net_data_path = os.path.join(predict_path, "GMnetKn1e1_dataV3")

Model = train_GM_Kn_Step4.load_model(os.path.join(net_data_path, CDFNet_file))

fNet = trainf_Kn.load_fmodel(os.path.join(net_data_path, fNet_file))


# we obtain the value q = S_w(w) by solving f(q) = w
def bisection_tensor(f, b_tensor, left_end, right_end, tol=1e-8, nmax=100000):
    left_end_tensor = torch.ones_like(b_tensor, requires_grad=True) * left_end
    right_end_tensor = torch.ones_like(b_tensor, requires_grad=True) * right_end
    # check f should have opposite sign at left and right end
    is_same_sign = (f(left_end_tensor) - b_tensor) * (f(right_end_tensor) - b_tensor)
    if torch.max(is_same_sign).detach().numpy() > 0.0:
        print("endpoint values have same sign: f(left_end)-b and f(right_end)-b")
        return

    n = 0
    while n <= nmax:

        # middle point
        mid_tensor = (left_end_tensor + right_end_tensor) / 2.0

        if torch.max(torch.abs(f(mid_tensor) - b_tensor)).detach().numpy() < tol or torch.max(
                right_end_tensor - left_end_tensor).detach().numpy() < tol:
            return mid_tensor.detach()

        is_same_sign = (f(mid_tensor) - b_tensor) * (f(left_end_tensor) - b_tensor) > 0.0
        left_end_tensor = mid_tensor * (is_same_sign) + left_end_tensor * (~is_same_sign)
        right_end_tensor = mid_tensor * (~is_same_sign) + right_end_tensor * (is_same_sign)

        n = n + 1
    else:
        print("exceed max iteration")


def func_f(x):
    zero_tensor = torch.zeros(x.shape)
    return fNet(x) - fNet(zero_tensor)


def S_w(w):
    return bisection_tensor(func_f, w, -230, 230)
# Kn = 1e-2 then (func_f, w, -0.1, 0.1) (-0.08, 0.08) (-0.15, 0.15):106+101 S+1013:(-0.4,0.4)
# Kn = 1e-1 then (func_f, w, -0.15, 0.15)  S+1013:(-0.4,0.4)
# kn = 1e0  then (func_f, w, -6, 1)
# kn = 1e1  then (func_f, w, -330, 330)
# kn = 1e1  then (func_f, w, -230, 230) 1e1_dataV3
# kn = 1e2  then (func_f, w, -33000, 33000) 1e2_dataV2

# ----------------------------

def equilibrium(u, v):
    vlength = len(v)
    xlength = len(u)
    f = torch.zeros([vlength, xlength])
    for i in range(vlength):
        f[i, :] = 0.5 * u
    return f

# ----------------------------
# Lax-Friedrich flux
# ----------------------------

def Flux(u, q):
    Theta_inv = 1 / u
    f1 = q
    f2 = Theta_inv
    return f1, f2


def LFscheme(u, w, n, T):
    L = 2 * math.pi
    vg = 1
    dx = L / n
    CFL = 0.01
    dt=dx*dx*0.1
    Nt = math.ceil(T / dt)
    # dt = T/Nt
    dt = T / Nt
    beta = dt / dx


    for i in range(Nt):
        # compute q = s_w
        w_out = w.reshape(n, 1)
        q_out = S_w(w_out)
        q_out = q_out.reshape(1, n)
        q = q_out[0]

        theta_inv = 1 / u
        cs = torch.sqrt(torch.abs(q))

        # U_{j+1} and U_{j-1} with periodic boundary
        u_p = torch.cat([u[1:], u[0:1]], dim=0)
        u_m = torch.cat([u[-1:], u[0:-1]], dim=0)
        w_p = torch.cat([w[1:], w[0:1]], dim=0)
        w_m = torch.cat([w[-1:], w[0:-1]], dim=0)
        q_p = torch.cat([q[1:], q[0:1]], dim=0)
        q_m = torch.cat([q[-1:], q[0:-1]], dim=0)

        # F_{j+1} and F_{j-1}
        F_p = Flux(u_p, q_p)
        F_m = Flux(u_m, q_m)

        u = u - 0.5 * beta * (F_p[0] - F_m[0]) \
              + 0.5 * (u_p - 2 * u + u_m)*dt/dx*0.1
        M0 = Model.M(u.reshape(n, 1), q.reshape(n, 1), Kn)
        M0 = M0.reshape(1, n)[0]
        Q = M0 * q
        w = w - 0.5 * beta * (F_p[1] - F_m[1])  + dt * Q\
             + 0.5 * (w_p - 2 * w + w_m)*dt/dx*0.1

    return u, w, q


# --------------------------
# Predictions
# --------------------------
refine_num = 1
Nx = 160
Nx_refine = refine_num * Nx

for init_num in predicition_group:

    for i_num in range(1, 2):

        for T_num in range(1, 11):

            T = 0.1 * T_num
            final_time = format(T, '.1f')

            print("predict group number: ", init_num, ";  initial data number: ", i_num, ";  final time: ", final_time)

            is_shock = True if (init_num > 100) else False

            init_data_path = os.path.join(matlab_path_data, str(init_num) + "st_init")

            init_data_file = os.path.join(init_data_path, str(init_num) + initial_Kn_name)

            u_1, q_1 = read_data(init_data_file, T_num, i_num)
            u_exact_np = u_1.detach().numpy()
            q_exact_np = q_1.detach().numpy()

            u_0, q_0 = read_data_inintial(init_data_file)
            w_out = func_f(q_0.reshape(Nx, 1))
            w_0 = w_out.reshape(1, Nx)[0]

            u_pred, w_pred, q_pred= LFscheme(u_0, w_0, Nx_refine, T)
            u_pred_np = u_pred[0::refine_num].detach().numpy()
            w_pred_np = w_pred[0::refine_num].detach().numpy()
            q_pred_np = q_pred[0::refine_num].detach().numpy()

            u_0_np = u_0.detach().numpy()
            u_0 = u_0[0::refine_num]
            # --------------------------
            # Figures and error
            # L2 and L1 error for CDF model
            Err_rho_L2 = np.sqrt(sum((u_pred_np - u_exact_np) ** 2) / sum(u_exact_np ** 2))
            Err_rho_L1 = sum(abs(u_pred_np - u_exact_np)) / sum(abs(u_exact_np))

            Err_rho_L2_q = np.sqrt(sum((q_pred_np - q_exact_np) ** 2) / sum(q_exact_np ** 2))
            Err_rho_L1_q = sum(abs(q_pred_np - q_exact_np)) / sum(abs(q_exact_np))

            print("L2_u error:{}".format(Err_rho_L2))
            print("L1_u error:{}".format(Err_rho_L1))

            print("L2_q error:{}".format(Err_rho_L2_q))
            print("L1_q error:{}".format(Err_rho_L1_q))

            result_path = os.path.join(curr_path, "results")
            result_num_path = os.path.join(result_path,
                                           "int" + str(init_num) + "_inum" + str(i_num) + "_Kn" + str(Kn) + "_T" + str(
                                               T))

            if not os.path.exists(result_num_path):
                os.makedirs(result_num_path)

            # plot and save figures
            dNx = 2 * pi / Nx
            x_np = np.linspace(-pi, pi - dNx, Nx)

            mksize = 2

            # for internal energy u
            plt.figure()
            label_predict = 'pre T = ' + str(T)
            label_exact = 'exact'
            label_init = 'initial condition'
            plt.plot(x_np, u_pred_np, 'o', markersize=mksize, label=label_predict)
            #plt.plot(x_np, w_pred_np, 'o', markersize=mksize, label=label_predict)
            plt.plot(x_np, u_exact_np, '-', label=label_exact)
            #plt.plot(x_np, q_pred_np, 's', markersize=mksize, label=label_predict)
            #plt.plot(x_np, q_exact_np, '-', label=label_exact)
            plt.title('internal energy $u$',fontsize=15)
            plt.legend(fontsize=12)
            plt.show()

            plt.figure()
            # label_predict = 'predict T = ' + str(T)
            label_predict = 'pre T = ' + str(T)
            label_exact = 'exact'
            label_init = 'initial condition'

            plt.plot(x_np, q_pred_np, 's', markersize=mksize, label=label_predict)
            plt.plot(x_np, q_exact_np, '-', label=label_exact)
            plt.title('heat flux $q$',fontsize=15)
            plt.legend(fontsize=12)
            plt.show()

            plt.figure()
            label_predict = 'T = ' + str(T)
            # label_exact = 'exact'
            label_init = 'initial condition '
            plt.plot(x_np, w_pred_np, '-', color='b', markersize=mksize, label=label_predict)
            plt.title('dissipative variable $w$',fontsize=15)
            plt.legend(fontsize=12)
            plt.show()

