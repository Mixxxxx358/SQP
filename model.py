from acados_template import AcadosModel
from casadi import SX, vertcat, sin, cos, Function, mtimes, tanh, MX
import deepSI
from torch import nn
import numpy as np

def export_ode_model() -> AcadosModel:

    model_name = 'unbalanced_disc_ode'

    # constants
    M = 0.0761844495320390 # mass of the cart [kg] -> now estimated
    g = 9.80155078791343 # gravity constant [m/s^2]
    J = 0.000244210523960356
    Km = 10.5081817407479
    I = 0.0410772235841364
    tau = 0.397973147009910

    # set up states & controls
    theta   = SX.sym('theta')
    dtheta  = SX.sym('dtheta')

    x = vertcat(dtheta, theta)

    F = SX.sym('F')
    u = vertcat(F)

    # dynamics
    f_expl = vertcat(-M*g*I/J*sin(theta) - 1/tau*dtheta + Km/tau*F, dtheta)
    h_disc = vertcat(theta, dtheta)
    # h_disc = vertcat(sin(theta), cos(theta), dtheta)
    # h_disc_e = vertcat(sin(theta), cos(theta), dtheta)

    model = AcadosModel()

    model.f_expl_expr = f_expl
    model.x = x
    model.u = u
    model.name = model_name
    model.cost_y_expr = h_disc
    model.cost_y_expr_e = h_disc

    return model

def CasADi_Fn(ss_enc, cas_x, cas_u):
    n_hidden_layers = 2#ss_enc.f_n_hidden_layers
    nu = ss_enc.nu if ss_enc.nu is not None else 1

    params = {}
    for name, param in ss_enc.fn.named_parameters():
        params[name] = param.detach().numpy()
    params_list = list(params.values())
    
    cas_xu = vertcat(cas_x,cas_u)

    temp_nn = cas_xu
    for i in range(n_hidden_layers):
        W_NL = params_list[2+i*2]
        b_NL = params_list[3+i*2]
        temp_nn = mtimes(W_NL, temp_nn)+b_NL
        temp_nn = tanh(temp_nn)
    W_NL = params_list[2+n_hidden_layers*2]
    b_NL = params_list[3+n_hidden_layers*2]
    nn_NL = mtimes(W_NL, temp_nn)+b_NL

    W_Lin = params_list[0]
    b_Lin = params_list[1]
    nn_Lin = mtimes(W_Lin,cas_xu) + b_Lin

    return nn_NL + nn_Lin

def CasADi_Hn(ss_enc, cas_x):
    n_hidden_layers = 2#ss_enc.h_n_hidden_layers

    params = {}
    for name, param in ss_enc.hn.named_parameters():
        params[name] = param.detach().numpy()
    params_list = list(params.values())

    temp_nn = cas_x
    for i in range(n_hidden_layers):
        W_NL = params_list[2+i*2]
        b_NL = params_list[3+i*2]
        temp_nn = mtimes(W_NL, temp_nn)+b_NL
        temp_nn = tanh(temp_nn)
    W_NL = params_list[2+n_hidden_layers*2]
    b_NL = params_list[3+n_hidden_layers*2]
    nn_NL = mtimes(W_NL, temp_nn)+b_NL

    W_Lin = params_list[0]
    b_Lin = params_list[1]
    nn_Lin = mtimes(W_Lin,cas_x) + b_Lin

    return nn_NL + nn_Lin


class SinCos_encoder(deepSI.fit_systems.SS_encoder_general):
    def __init__(self, nx=10, na=20, nb=20, na_right=0, nb_right=0, e_net_kwargs={}, f_net_kwargs={}, h_net_kwargs={}):
        super(SinCos_encoder, self).__init__(nx=nx, na=na, nb=nb, na_right=na_right, nb_right=nb_right, e_net_kwargs=e_net_kwargs, f_net_kwargs=f_net_kwargs, h_net_kwargs=h_net_kwargs)

    def init_nets(self, nu, ny): # a bit weird
        na_right = self.na_right if hasattr(self,'na_right') else 0
        nb_right = self.nb_right if hasattr(self,'nb_right') else 0
        self.encoder = self.e_net(nb=(self.nb+nb_right), nu=nu, na=(self.na+na_right), ny=ny, nx=self.nx, **self.e_net_kwargs)
        self.fn =      self.f_net(nx=self.nx, nu=nu,                                **self.f_net_kwargs)
        self.hn =      nn.Identity(self.nx)

    def init_model(self, sys_data=None, nu=-1, ny=-1, device='cpu', auto_fit_norm=True, optimizer_kwargs={}, parameters_optimizer_kwargs={}, scheduler_kwargs={}):
        '''This function set the nu and ny, inits the network, moves parameters to device, initilizes optimizer and initilizes logging parameters'''
        if sys_data==None:
            assert nu!=-1 and ny!=-1, 'either sys_data or (nu and ny) should be provided'
            self.nu, self.ny = nu, ny
        else:
            self.nu, self.ny = sys_data.nu, sys_data.ny
            if auto_fit_norm:
                self.norm.fit(sys_data)
        self.init_nets(self.nu, self.ny)
        self.to_device(device=device)
        parameters_and_optim = [{**item,**parameters_optimizer_kwargs.get(name,{})} for name,item in self.parameters_with_names.items()]
        self.optimizer = self.init_optimizer(parameters_and_optim, **optimizer_kwargs)
        self.scheduler = self.init_scheduler(**scheduler_kwargs)
        self.bestfit = float('inf')
        self.Loss_val, self.Loss_train, self.batch_id, self.time, self.epoch_id = np.array([]), np.array([]), np.array([]), np.array([]), np.array([])
        self.init_model_done = True

def export_casadi_fn(system) -> AcadosModel:
    nx = system.nx
    x = MX.sym("x",nx,1)
    nu = system.nu if system.nu is not None else 1
    u = MX.sym("u",nu,1)
    ny = system.ny if system.ny is not None else 1

    f_expr = CasADi_Fn(system, x, u)
    h_expr = CasADi_Hn(system, x)

    model = AcadosModel()

    # model.f_expl_expr = f_expr
    model.x = x
    model.u = u
    model.name = 'unbalanced_disc_fn'
    model.disc_dyn_expr = f_expr
    model.cost_y_expr = vertcat(h_expr, u)
    model.cost_y_expr_e = vertcat(h_expr)

    return model

if __name__ == '__main__':
    # model_ode = export_ode_model()

    system = deepSI.load_system('trained_models/identity_dt0_1_e100_b1000_nf25_amp3_0_sn0_014')
    model_cas = export_casadi_fn(system)