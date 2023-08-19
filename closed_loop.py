from acados_template import AcadosOcp, AcadosOcpSolver, AcadosSimSolver
from model import export_ode_model, export_casadi_fn
from utils import *
import numpy as np
import scipy.linalg
from model import SinCos_encoder
import deepSI
from systems import UnbalancedDisc, ReversedUnbalancedDisc
import time

def main():
    # create ocp object to formulate the OCP
    ocp = AcadosOcp()

    # set model
    # ss_enc = deepSI.load_system('trained_models/identity_dt0_1_e100_b1000_nf25_amp3_0_sn0_014')
    # model = export_casadi_fn(ss_enc)
    model = export_ode_model()
    ocp.model = model

    # norm = ss_enc.norm
    norm = normalizer(np.array([0.0,0.0]), np.array([1.0,1.0]), 0, 1)

    # nx = model.x.size()[0]
    # nu = model.u.size()[0]
    nx = 2
    nu = 1
    ny = nx + nu
    ny_e = nx
    N_horizon = 5
    dt = 0.1
    Tf = dt*N_horizon

    # set dimensions
    ocp.dims.N = N_horizon

    # set cost module
    ocp.cost.cost_type = 'NONLINEAR_LS'
    ocp.cost.cost_type_e = 'NONLINEAR_LS'

    Q_mat = 2*np.diag([1e2, 1e0])
    R_mat = 2*np.diag([1e-2])

    ocp.cost.W = Q_mat#scipy.linalg.block_diag(Q_mat, R_mat)

    ocp.cost.W_e = Q_mat

    # ocp.cost.Vx = np.zeros((ny, nx))
    # ocp.cost.Vx[:nx,:nx] = np.eye(nx)

    # Vu = np.zeros((ny, nu))
    # Vu[2,0] = 1.0
    # ocp.cost.Vu = Vu

    # ocp.cost.Vx_e = np.eye(nx)

    theta_ref = 0
    ocp.cost.yref  = np.hstack((norm_output((theta_ref,0), norm)))#np.zeros((ny, ))
    ocp.cost.yref_e = np.hstack((norm_output((theta_ref,0), norm)))#np.zeros((ny_e, ))


    # set constraints
    Fmax = 4.0
    x0 = norm_output(np.array([0.0, 0.0]), norm)
    ocp.constraints.constr_type = 'BGH'
    ocp.constraints.lbu = norm_input(np.array([-Fmax]), norm)
    ocp.constraints.ubu = norm_input(np.array([+Fmax]), norm)
    # ocp.constraints.lbx = norm_output(np.array([-1.5, -10]), norm)
    # ocp.constraints.ubx = norm_output(np.array([1.5, 10]), norm)
    ocp.constraints.x0 = x0
    ocp.constraints.idxbu = np.array([0])
    # ocp.constraints.idxbx = np.array([0,0]) # look at this later

    # ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM'
    ocp.solver_options.qp_solver = 'FULL_CONDENSING_QPOASES'
    ocp.solver_options.hessian_approx = 'GAUSS_NEWTON'
    ocp.solver_options.integrator_type = 'ERK'
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
    a = 1.1; reference = np.hstack((np.ones(20)*a,np.ones(20)*-a,np.ones(20)*a/2,np.ones(20)*-a/2,np.ones(40)*0))
    # reference = np.load("references/multisine.npy")
    # a=3.1; reference = []
    # for i in range(40):
    #     reference = np.hstack((reference, np.ones(5)*a/40*i))
    # reference = np.hstack((reference, np.ones(40)*a))

    simX = np.ndarray((Nsim+1, nx))
    simU = np.ndarray((Nsim, nu))

    simX[0,:] = denorm_output(x0, norm)
    total_time = 0

    comp_time_total = np.zeros(Nsim)
    solver_time_total = np.zeros(Nsim)
    lin_time_total = np.zeros(Nsim)

    start_time = time.time()
    
    # closed loop
    for i in range(Nsim):
        for j in range(1,N_horizon):
            acados_ocp_solver.cost_set(j, 'yref', np.hstack((norm_output((reference[i+j],0), norm))))
        acados_ocp_solver.cost_set(N_horizon, 'yref', np.hstack((norm_output((reference[i+N_horizon],0), norm))))

        # solve ocp and get next control input
        simU[i,:] = denorm_input(acados_ocp_solver.solve_for_x0(x0_bar = norm_output(simX[i, :], norm)), norm)
        # acados_ocp_solver.print_statistics()
        total_time += acados_ocp_solver.get_stats('time_tot')
        comp_time_total[i] = acados_ocp_solver.get_stats('time_tot')
        solver_time_total[i] = acados_ocp_solver.get_stats('time_qp_solver_call')
        lin_time_total[i] = acados_ocp_solver.get_stats('time_lin')
        # print(acados_ocp_solver.get_stats('sqp_iter'))

        system.x = system.f(system.x, simU[i,0])
        x1 = system.h(system.x, simU[i,0])

        simX[i+1, :] = x1

        
        # simulate system
        # noise = np.random.normal(0, 0.03, size=(2,))
        # simX[i+1, :] = acados_integrator.simulate(x=simX[i, :], u=simU[i,:]) + noise

    end_time = time.time()
    print("Sim time: " + str(end_time - start_time) + " [s]")

    # plot results
    # simX[:, 1] = (simX[:, 1]+np.pi)%(2*np.pi) - np.pi
    reference = (reference+np.pi)%(2*np.pi) - np.pi
    print("Total time: " + str(total_time[0]) + " [s]")
    print("Average time per step: " + str(total_time[0]/Nsim * 1000) + " [ms]")
    plot_pendulum(np.linspace(0, Tf/N_horizon*Nsim, Nsim+1), Fmax, simU, simX, reference)

    # np.save("experiments/ud_sqp_levels_u", simU)
    # np.save("experiments/ud_sqp_levels_q", simX)

    # np.save("experiments/ud_sqp_sinus_u", simU)
    # np.save("experiments/ud_sqp_sinus_q", simX)

    print([np.mean(comp_time_total)*1000, np.max(comp_time_total)*1000, np.std(comp_time_total)*1000, np.mean(solver_time_total)*1000, np.mean(lin_time_total)*1000])


if __name__ == '__main__':
    main()