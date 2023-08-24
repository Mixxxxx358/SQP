from acados_template import AcadosOcp, AcadosOcpSolver, AcadosSimSolver
from model import export_ode_model, export_casadi_fn
from utils import *
import numpy as np
import scipy.linalg
from model import SinCos_encoder
import deepSI
from systems import UnbalancedDisc, ReversedUnbalancedDisc
import time
import torch

def main():
    # create ocp object to formulate the OCP
    ocp = AcadosOcp()

    # set model
    # ss_enc = deepSI.load_system('trained_models/identity_dt0_1_e100_b1000_nf25_amp3_0_sn0_014')
    ss_enc = deepSI.load_system('trained_models/ObserverUnbalancedDisk_dt01_nab_4_SNR_30_e250')
    model = export_casadi_fn(ss_enc)
    # model = export_ode_model()
    ocp.model = model

    norm = ss_enc.norm
    # norm = normalizer(np.array([0.0,0.0]), np.array([1.0,1.0]), 0, 1)

    nx = model.x.size()[0]
    nu = model.u.size()[0]
    ny = 1
    N_horizon = 5
    dt = 0.1
    Tf = dt*N_horizon

    # set dimensions
    ocp.dims.N = N_horizon

    # set cost module
    ocp.cost.cost_type = 'NONLINEAR_LS'
    ocp.cost.cost_type_e = 'NONLINEAR_LS'

    Q_mat = 2*np.diag([1e2,1e1])
    Q_e = 2*np.diag([1e2])

    ocp.cost.W = Q_mat
    ocp.cost.W_e = Q_e

    ocp.cost.yref  = np.zeros((ny+nu, )) #norm_output((theta_ref), norm)
    ocp.cost.yref_e = np.zeros((ny, ))#np.zeros((ny_e, ))


    # set constraints
    Fmax = 4.0
    x0 = norm_output(np.array([0.0, 0.0]), norm)
    ocp.constraints.constr_type = 'BGH'
    ocp.constraints.lbu = norm_input(np.array([-Fmax]), norm)
    ocp.constraints.ubu = norm_input(np.array([+Fmax]), norm)
    ocp.constraints.x0 = x0
    ocp.constraints.idxbu = np.array([0])

    # ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM'
    ocp.solver_options.qp_solver = 'FULL_CONDENSING_QPOASES'
    ocp.solver_options.hessian_approx = 'GAUSS_NEWTON'
    ocp.solver_options.integrator_type = 'DISCRETE'
    ocp.solver_options.nlp_solver_type = 'SQP_RTI' # SQP_RTI

    ocp.solver_options.qp_solver_cond_N = N_horizon

    # set prediction horizon
    ocp.solver_options.tf = Tf

    solver_json = 'acados_ocp_' + model.name + '.json'
    acados_ocp_solver = AcadosOcpSolver(ocp, json_file = solver_json)

    # create an integrator with the same settings as used in the OCP solver.
    # acados_integrator = AcadosSimSolver(ocp, json_file = solver_json)

    system = UnbalancedDisc(dt=dt, sigma_n=[0.0])
    system.reset_state()

    Nsim = 100
    # a = 0.9; reference = np.hstack((np.ones(20)*a,np.ones(20)*-a,np.ones(20)*a/2,np.ones(20)*-a/2,np.ones(40)*0))
    reference = np.load("references/multisine.npy")
    # reference = randomLevelReference(Nsim+N_horizon, [10,15], [-1.1,1.1])
    # a = 0.; reference = np.hstack((np.ones(20)*a,np.ones(20)*-a,np.ones(20)*a/2,np.ones(20)*-a/2,np.ones(40)*0))
    # reference = np.load("references/setPoints.npy")
    # reference = np.hstack((np.ones(200)*0.2))

    simX = np.ndarray((Nsim+1, nx))
    simU = np.ndarray((Nsim, nu))

    total_time = 0
    log_time = np.zeros(Nsim)
    total_solver_time = 0
    total_linear_time = 0

    u0 = 0
    y0 = np.zeros(1,)
    nb = ss_enc.nb
    uhist = torch.ones((1,nb))*u0
    na = ss_enc.na
    yhist = torch.Tensor(y0[np.newaxis].T).repeat(1,na+1)[None,:,:]

    simX[0,:] = ss_enc.encoder(uhist,yhist).detach().numpy()

    # ocp.constraints.x0 = simX[0,:].T
    # print(simX[0,:].shape)
    
    log_y = np.zeros(Nsim)

    start_time = time.time()

    # closed loop
    for i in range(Nsim):
        for j in range(1,N_horizon):
            acados_ocp_solver.cost_set(j, 'yref', norm_output(np.hstack((reference[i+j],setPointInput(reference[i+j]))),norm))
        acados_ocp_solver.cost_set(N_horizon, 'yref', norm_output((reference[i+j]),norm))

        # solve ocp and get next control input
        simU[i,:] = denorm_input(acados_ocp_solver.solve_for_x0(x0_bar = simX[i, :]), norm)
        # acados_ocp_solver.print_statistics()
        time_step = acados_ocp_solver.get_stats('time_tot')
        total_time += time_step
        log_time[i] = time_step
        total_solver_time += acados_ocp_solver.get_stats('time_qp')
        total_linear_time += acados_ocp_solver.get_stats('time_lin')
        # print(acados_ocp_solver.get_stats('sqp_iter'))

        system.x = system.f(system.x, simU[i,0])
        x1 = system.h(system.x, simU[i,0])
        y1 = x1[1]

        # shift history input and output for encoder
        for j in range(nb-1):
            uhist[0,j] = uhist[0,j+1]
        uhist[0,nb-1] = torch.Tensor(norm_input(simU[i,:],norm))
        for j in range(na):
            yhist[0,:,j] = yhist[0,:,j+1]
        yhist[0,:,na] = torch.Tensor([norm_output(y1,norm)])
        # predict state with encoder
        x1 = ss_enc.encoder(uhist,yhist).detach().numpy().T

        simX[i+1, :] = x1[:,0]

        log_y[i] = y1

    end_time = time.time()
    print("Sim time: " + str(end_time - start_time) + " [s]")

    # plot results
    # simX[:, 1] = (simX[:, 1]+np.pi)%(2*np.pi) - np.pi
    reference = (reference+np.pi)%(2*np.pi) - np.pi
    print("Total time: " + str(total_time[0]) + " [s]")
    print("Average time per step: " + str(total_time[0]/Nsim * 1000) + " [ms]")
    print("Std dev: " + str(np.std(log_time)*1000) + " [ms]")
    print("Total linear time: " + str(total_linear_time[0]) + " [s]")
    print("Average linear time per step: " + str(total_linear_time[0]/Nsim * 1000) + " [ms]")
    print("Total solver time: " + str(total_solver_time[0]) + " [s]")
    print("Average solver time per step: " + str(total_solver_time[0]/Nsim * 1000) + " [ms]")


    plt.subplot(2,1,1)
    plt.plot(simU[:,0])
    plt.plot(np.ones(Nsim)*Fmax, 'r-.')
    plt.plot(np.ones(Nsim)*-Fmax, 'r-.')
    plt.grid()

    plt.subplot(2,1,2)
    plt.plot(log_y)
    plt.plot(reference[:Nsim], "k--")
    plt.grid()

    plt.show()
    plt.savefig("test.png")


    # plot_pendulum(np.linspace(0, Tf/N_horizon*Nsim, Nsim+1), Fmax, simU, simX, reference)

    # np.save("experiments/ud_sqp_encoder_levels_u", simU)
    # np.save("experiments/ud_sqp_encoder_levels_q", log_y)

    np.save("experiments/ud_sqp_encoder_sinus_u", simU)
    np.save("experiments/ud_sqp_encoder_sinus_q", log_y)

if __name__ == '__main__':
    main()