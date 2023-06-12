from acados_template import AcadosOcp, AcadosOcpSolver, AcadosSimSolver
from model import export_ode_model, export_casadi_fn
from utils import *
import numpy as np
import scipy.linalg
from model import SinCos_encoder
import deepSI
from systems import UnbalancedDisc, ReversedUnbalancedDisc

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

    nx = 2#model.x.size()[0]
    nu = 1#model.u.size()[0]
    ny = nx + nu
    ny_e = nx
    N_horizon = 15
    dt = 0.1
    Tf = dt*N_horizon

    # set dimensions
    ocp.dims.N = N_horizon

    # set cost module
    ocp.cost.cost_type = 'NONLINEAR_LS'
    ocp.cost.cost_type_e = 'NONLINEAR_LS'

    Q_mat = 2*np.diag([1e2, 1e2, 1e-2])
    R_mat = 2*np.diag([1e-1])

    ocp.cost.W = Q_mat#scipy.linalg.block_diag(Q_mat, R_mat)

    ocp.cost.W_e = Q_mat

    # ocp.cost.Vx = np.zeros((ny, nx))
    # ocp.cost.Vx[:nx,:nx] = np.eye(nx)

    # Vu = np.zeros((ny, nu))
    # Vu[2,0] = 1.0
    # ocp.cost.Vu = Vu

    # ocp.cost.Vx_e = np.eye(nx)

    theta_ref = 0
    ocp.cost.yref  = np.hstack((norm_output((np.sin(theta_ref), np.cos(theta_ref)), norm),0))#np.zeros((ny, ))
    ocp.cost.yref_e = np.hstack((norm_output((np.sin(theta_ref), np.cos(theta_ref)), norm),0))#np.zeros((ny_e, ))


    # set constraints
    Fmax = 2.5
    x0 = norm_output(np.array([0.0, 0.0]), norm)
    ocp.constraints.constr_type = 'BGH'
    ocp.constraints.lbu = norm_input(np.array([-Fmax]), norm)
    ocp.constraints.ubu = norm_input(np.array([+Fmax]), norm)
    # ocp.constraints.lbx = norm_output(np.array([-1.5, -10]), norm)
    # ocp.constraints.ubx = norm_output(np.array([1.5, 10]), norm)
    ocp.constraints.x0 = x0
    ocp.constraints.idxbu = np.array([0])
    # ocp.constraints.idxbx = np.array([0,0]) # look at this later

    ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM' # FULL_CONDENSING_QPOASES
    ocp.solver_options.hessian_approx = 'GAUSS_NEWTON'
    ocp.solver_options.integrator_type = 'ERK'
    ocp.solver_options.nlp_solver_type = 'SQP' # SQP_RTI

    ocp.solver_options.qp_solver_cond_N = N_horizon

    # set prediction horizon
    ocp.solver_options.tf = Tf

    solver_json = 'acados_ocp_' + model.name + '.json'
    acados_ocp_solver = AcadosOcpSolver(ocp, json_file = solver_json)

    # create an integrator with the same settings as used in the OCP solver.
    acados_integrator = AcadosSimSolver(ocp, json_file = solver_json)

    system = UnbalancedDisc(dt=dt, sigma_n=[0.014])
    system.reset_state()

    Nsim = 50
    # reference = randomLevelReference(Nsim+N_horizon, [10,15], [-np.pi,np.pi])
    reference = np.hstack((np.ones(200)*2.8))

    simX = np.ndarray((Nsim+1, nx))
    simU = np.ndarray((Nsim, nu))

    simX[0,:] = denorm_output(x0, norm)
    total_time = 0
    
    # closed loop
    for i in range(Nsim):
        for j in range(1,N_horizon):
            acados_ocp_solver.cost_set(j, 'yref', np.hstack((norm_output((np.sin(reference[i+j]), np.cos(reference[i+j])), norm),0)))
        acados_ocp_solver.cost_set(N_horizon, 'yref', np.hstack((norm_output((np.sin(reference[i+N_horizon]), np.cos(reference[i+N_horizon])), norm),0)))

        # solve ocp and get next control input
        simU[i,:] = denorm_input(acados_ocp_solver.solve_for_x0(x0_bar = norm_output(simX[i, :], norm)), norm)
        total_time += acados_ocp_solver.get_stats('time_tot')

        system.x = system.f(system.x, simU[i,0])
        x1 = system.h(system.x, simU[i,0])

        simX[i+1, :] = x1

        
        # simulate system
        # noise = np.random.normal(0, 0.03, size=(2,))
        # simX[i+1, :] = acados_integrator.simulate(x=simX[i, :], u=simU[i,:]) + noise

    # plot results
    simX[:, 1] = (simX[:, 1]+np.pi)%(2*np.pi) - np.pi
    reference = (reference+np.pi)%(2*np.pi) - np.pi
    print(total_time)
    plot_pendulum(np.linspace(0, Tf/N_horizon*Nsim, Nsim+1), Fmax, simU, simX, reference)


if __name__ == '__main__':
    main()