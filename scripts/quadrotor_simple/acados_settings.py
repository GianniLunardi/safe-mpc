#
# Copyright (c) The acados authors.
#
# This file is part of acados.
#
# The 2-Clause BSD License
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice,
# this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.;
#

# reference : "Towards Time-optimal Tunnel-following for Quadrotors", Jon Arrizabalaga et al.

import casadi as ca
from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver, AcadosSimSolver
import scipy.linalg

from common import *
from sys_dynamics import SysDyn

class AcadosCustomOcp:

    def __init__(self):
        self.nx = 0
        self.nu = 0
        self.ny = 0
        self.ns = 0

        self.ocp = None,
        self.solver = None,
        self.integrator = None
        self.sysModel = None

        self.zeta_0 = np.array([0.05, 0, 0,       # x,  y,  z
                      1, 0, 0, 0,       # qw, qx, qy, qz,
                      0, 0, 0,          # ohmr,  ohmp,  ohmy
                      0, 0, 0,  ])  
        self.zeta_N = None
        self.u_N = None


    def setup_acados_ocp(self, params, model, nl_cost=True):
        '''Formulate Acados OCP'''

        

        # create Acados model
        ocp = AcadosOcp()
        ocp.model = model.amodel

        # set dimensions
        ocp.solver_options.N_horizon = N
        self.nx = model.amodel.x.size()[0]
        print(self.nx)
        self.nu = model.amodel.u.size()[0]
        ny = self.nx + self.nu

        self.zeta_N = ca.repmat(np.reshape(self.zeta_0, (self.nx,1)), 1, N+1)
        self.u_N = ca.repmat(U_REF, 1, N)

        # continuity constraints
        ocp.constraints.x0  = self.zeta_0

        # ocp.constraints.lbx_0 = -1e6*np.ones(model.x_min.shape[0]) + model.x_min + (model.params.q_margin/100) * model.bounds_diff
        # ocp.constraints.ubx_0 = 1e6*np.ones(model.x_min.shape[0]) + model.x_max - (model.params.q_margin/100) * model.bounds_diff
        # ocp.constraints.idxbx_0 = np.arange(model.nx)

        # formulate cost function
        if nl_cost:
            ocp.cost.cost_type = "NONLINEAR_LS"
            ocp.model.cost_y_expr = ca.vertcat(model.amodel.x, model.amodel.u)
            ocp.cost.yref = np.array([ 0.2, 0, 0,
                                    1, 0, 0, 0,
                                    0, 0, 0,
                                    0, 0, 0,
                                    #   U_HOV, U_HOV, U_HOV, U_HOV,
                                    0, 0, 0, 0])
            ocp.cost.W = scipy.linalg.block_diag(Q, R)

            ocp.cost.cost_type_e = "NONLINEAR_LS"
            ocp.model.cost_y_expr_e = model.amodel.x
            ocp.cost.yref_e = np.array([ 0.2, 0, 0,
                                        1, 0, 0, 0,
                                        0, 0, 0,
                                        0, 0, 0,])
                                        #  U_HOV, U_HOV, U_HOV, U_HOV])
            ocp.cost.W_e = Qn

        else:
            ocp.cost.cost_type = "LINEAR_LS"
            ocp.cost.W = scipy.linalg.block_diag(np.diag(params.Q_weight), np.diag(params.R_weight))
            ocp.cost.W_e = np.diag(params.Qn_weight)

            ocp.cost.yref = np.zeros(ny)
            ocp.cost.yref_e = np.zeros(self.nx)

            ocp.cost.Vx = np.zeros((ny, self.nx))
            ocp.cost.Vx[:self.nx, :self.nx] = np.eye(self.nx)

            ocp.cost.Vx_e = np.eye(self.nx)

            ocp.cost.Vu = np.zeros((ny, self.nu))
            ocp.cost.Vu[-self.nu:, :] = np.eye(self.nu)


        # formulate inquality constraints

        # constrain AGV dynamics : acceleration, angular velocity (convex ?, Non-linear)
        dyn_constr_eqn = []

        ineq_constr_eqn = []
        ineq_constr_eqn = ca.vertcat(ineq_constr_eqn, dyn_constr_eqn)

        model.con_h_expr = ineq_constr_eqn
        model.con_h_expr_e = ineq_constr_eqn

        # inequality bounds
        nh = model.con_h_expr.shape[0]

        # constrain controls
        lbu = [0] * self.nu
        ubu = [U_MAX] * self.nu

        # # Control bounds ( Affects horizon quality before switch)
        # lbu[0] = OHM_MIN;         ubu[0] = OHM_MAX
        # lbu[1] = OHM_MIN;         ubu[1] = OHM_MAX

        ocp.constraints.lbu = np.array(lbu)
        ocp.constraints.ubu = np.array(ubu)
        ocp.constraints.idxbu = np.array([0, 1, 2, 3])

        # Bounds on path constraints (inequality)
        lh = np.zeros(nh);        uh = np.zeros(nh)
        lh[:] = -INF;             uh[:] = INF

        ocp.constraints.lh = lh
        ocp.constraints.uh = uh

        ocp.constraints.lh_e = lh
        ocp.constraints.uh_e = uh

        # configure itegrator and QP solver
        ocp.solver_options.integrator_type = "ERK"
        ocp.solver_options.tf = Tf
        ocp.solver_options.sim_method_num_stages = 4
        ocp.solver_options.sim_method_num_steps = 1
        # ocp.solver_options.collocation_type = 'GAUSS_RADAU_IIA'
        # ocp.solver_options.time_steps = time_steps
        # ocp.solver_options.shooting_nodes = shooting_nodes

        ocp.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM"#"PARTIAL_CONDENSING_HPIPM" #"FULL_CONDENSING_HPIPM" #"PARTIAL_CONDENSING_HPIPM"
        ocp.solver_options.hessian_approx =  "GAUSS_NEWTON"#"EXACT",
        # ocp.solver_options.cost_discretization ="INTEGRATOR"
        ocp.solver_options.qp_solver_cond_N = int(N/2)
        ocp.solver_options.nlp_solver_type = "SQP"
        # ocp.solver_options.tol = 1e-3
        # ocp.qp_solver_tol = 1e-3

        # create solver
        solve_json = "planner_ocp.json"
        self.ocp = ocp
        self.solver = AcadosOcpSolver(ocp, json_file = solve_json)
        self.integrator = AcadosSimSolver(ocp, json_file = solve_json) #TODO

        return True
    
    def my_ocp_setup(self, params, model):
        '''Formulate Acados OCP'''

        # create Acados model
        ocp = AcadosOcp()
        ocp.model = model.amodel

        # set dimensions
        ocp.solver_options.N_horizon = N
        self.nx = model.amodel.x.size()[0]
        self.nu = model.amodel.u.size()[0]
        ny = self.nx + self.nu

        self.x_guess = ca.repmat(np.reshape(self.zeta_0, (self.nx,1)), 1, N+1)
        self.u_guess = ca.repmat(U_REF, 1, N)

        # continuity constraints
        ocp.constraints.x0  = self.zeta_0

        
        ocp.cost.cost_type = "LINEAR_LS"
        ocp.cost.W = scipy.linalg.block_diag(np.diag(params.Q_weight), np.diag(params.R_weight))
        ocp.cost.W_e = np.diag(params.Qn_weight)

        ocp.cost.yref = np.zeros(ny)
        ocp.cost.yref_e = np.zeros(self.nx)

        ocp.cost.Vx = np.zeros((ny, self.nx))
        ocp.cost.Vx[:self.nx, :self.nx] = np.eye(self.nx)

        ocp.cost.Vx_e = np.eye(self.nx)

        ocp.cost.Vu = np.zeros((ny, self.nu))
        ocp.cost.Vu[-self.nu:, :] = np.eye(self.nu)


        # formulate inquality constraints

        # constrain AGV dynamics : acceleration, angular velocity (convex ?, Non-linear)
        dyn_constr_eqn = []

        ineq_constr_eqn = []
        ineq_constr_eqn = ca.vertcat(ineq_constr_eqn, dyn_constr_eqn)

        model.con_h_expr = ineq_constr_eqn
        model.con_h_expr_e = ineq_constr_eqn

        # inequality bounds
        nh = model.con_h_expr.shape[0]

        # constrain controls
        lbu = [0] * self.nu
        ubu = [U_MAX] * self.nu

        
        ocp.constraints.lbu = np.array(lbu)
        ocp.constraints.ubu = np.array(ubu)
        ocp.constraints.idxbu = np.array([0, 1, 2, 3])

        ocp.constraints.lbx = np.array([-0.05, -0.05, -1.3])
        ocp.constraints.ubx = np.array([INF, INF, INF])
        ocp.constraints.idxbx = np.array([0, 1, 2])

        # Bounds on path constraints (inequality)
        lh = np.zeros(nh)        
        uh = np.zeros(nh)
        lh[:] = -INF            
        uh[:] = INF

        ocp.constraints.lh = lh
        ocp.constraints.uh = uh

        ocp.constraints.lh_e = lh
        ocp.constraints.uh_e = uh

        # configure itegrator and QP solver
        ocp.solver_options.integrator_type = "ERK"
        ocp.solver_options.tf = Tf
        ocp.solver_options.sim_method_num_stages = 4
        ocp.solver_options.sim_method_num_steps = 1
        
        ocp.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM"#"PARTIAL_CONDENSING_HPIPM" #"FULL_CONDENSING_HPIPM" #"PARTIAL_CONDENSING_HPIPM"
        ocp.solver_options.hessian_approx =  "GAUSS_NEWTON"#"EXACT",
        ocp.solver_options.qp_solver_cond_N = int(N/2)
        ocp.solver_options.nlp_solver_type = "SQP"
        
        # create solver
        solve_json = "planner_ocp.json"
        self.ocp = ocp
        self.solver = AcadosOcpSolver(ocp, json_file = solve_json)
        
        return True


    def solve_and_sim(self, model):
        '''Solve the OCP with multiple shooting, and forward simulate with RK4'''

        # Integrate ODE model to get CL estimate (no measurement noise)
        u_0 = self.solver.solve_for_x0(self.zeta_0)
        self.zeta_0 = model.acados_integrator.simulate(x=self.zeta_0, u=u_0)

        # u_0 = self.solver.solve_for_x0(self.zeta_0)
        self.x_guess = np.reshape(self.solver.get(0, "x"), (self.nx, 1))
        # self.zeta_0 = self.solver.get(0, "x")
        for i in range(1, N +1):
            zeta_i = np.reshape(self.solver.get(i, "x"), (self.nx, 1))
            self.x_guess = np.concatenate((self.x_guess, zeta_i), axis = 1)

        self.u_guess[:, 0] = u_0
        # return status

    def solve(self):
        # self.solver.reset()
        self.solver.set(0, 'lbx', self.zeta_0)
        self.solver.set(0, 'ubx', self.zeta_0)

        # for i in range(N):
        #     print(self.x_guess[i])
        #     self.solver.set(i, 'x', self.x_guess[i])
        #     self.solver.set(i, 'u', self.u_guess[i])
        # self.solver.set(N, 'x', self.x_guess[-1])

        self.solver.solve()
        self.u_guess[:, 0] = self.solver.get(0, "u")


    def cost_update_ref(self, target):

        if np.linalg.norm(self.zeta_0[:3] - target) < 0.01:
            return True
        
        for j in range(N):
            
            # yref = np.array([target[0], target[1], target[2], 0, 0, 0, 0, 0, 0, 0, 0, 0., 0., U_HOV, U_HOV, U_HOV, U_HOV, 0, 0, 0, 0])
            yref = np.array([target[0], target[1], target[2], 0, 0, 0, 0, 0, 0, 0, 0, 0., 0.,  0, 0, 0, 0])
            self.solver.set(j, "yref", yref)

        # yref = np.array([target[0], target[1], target[2], 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, U_HOV, U_HOV, U_HOV, U_HOV])
        yref = np.array([target[0], target[1], target[2], 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        self.solver.set(N, "yref", yref)


    def get_cost(self):
        cost = self.solver.get_cost()
        return cost

