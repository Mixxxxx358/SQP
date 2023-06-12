from acados_template import AcadosOcp, AcadosOcpSolver
from model import export_ode_model, export_casadi_fn
import numpy as np
from utils import plot_pendulum
from model import SinCos_encoder
import deepSI

def main():
    # create ocp object to formulate the OCP
    ocp = AcadosOcp()

    # set model
    system = deepSI.load_system('trained_models/identity_dt0_1_e100_b1000_nf25_amp3_0_sn0_014')
    model = export_casadi_fn(system)
    ocp.model = model

    nx = model.x.size()[0]
    nu = model.u.size()[0]
    N = 10
    Tf = 0.1*N

    # set dimensions
    ocp.dims.N = N

    # set cost
    Q_mat = 2*np.diag([1e3, 1e-2])
    R_mat = 2*np.diag([1e-2])

    # the 'EXTERNAL' cost type can be used to define general cost terms
    # NOTE: This leads to additional (exact) hessian contributions when using GAUSS_NEWTON hessian.
    ocp.cost.cost_type = 'EXTERNAL'
    ocp.cost.cost_type_e = 'EXTERNAL'
    ocp.model.cost_expr_ext_cost = model.x.T @ Q_mat @ model.x + model.u.T @ R_mat @ model.u
    ocp.model.cost_expr_ext_cost_e = model.x.T @ Q_mat @ model.x

    # set constraints
    Fmax = 2
    ocp.constraints.lbu = np.array([-Fmax])
    ocp.constraints.ubu = np.array([+Fmax])
    ocp.constraints.idxbu = np.array([0])

    ocp.constraints.x0 = np.array([0.0, 0.0])
    ocp.cost.yref = np.array([0.5, 0.0])

    # set options
    ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM' # FULL_CONDENSING_QPOASES
    # PARTIAL_CONDENSING_HPIPM, FULL_CONDENSING_QPOASES, FULL_CONDENSING_HPIPM,
    # PARTIAL_CONDENSING_QPDUNES, PARTIAL_CONDENSING_OSQP, FULL_CONDENSING_DAQP
    ocp.solver_options.hessian_approx = 'GAUSS_NEWTON' # 'GAUSS_NEWTON', 'EXACT'
    ocp.solver_options.integrator_type = 'DISCRETE'
    # ocp.solver_options.print_level = 1
    ocp.solver_options.nlp_solver_type = 'SQP' # SQP_RTI, SQP

    # set prediction horizon
    ocp.solver_options.tf = Tf

    ocp_solver = AcadosOcpSolver(ocp, json_file = 'acados_ocp.json')

    simX = np.ndarray((N+1, nx))
    simU = np.ndarray((N, nu))

    status = ocp_solver.solve()
    ocp_solver.print_statistics() # encapsulates: stat = ocp_solver.get_stats("statistics")

    if status != 0:
        raise Exception(f'acados returned status {status}.')

    # get solution
    for i in range(N):
        simX[i,:] = ocp_solver.get(i, "x")
        simU[i,:] = ocp_solver.get(i, "u")
    simX[N,:] = ocp_solver.get(N, "x")

    plot_pendulum(np.linspace(0, Tf, N+1), Fmax, simU, simX, latexify=False)


if __name__ == '__main__':
    main()