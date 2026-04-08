import time
import pickle
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from scipy.stats import qmc
from safe_mpc.parser import Parameters, parse_args
from safe_mpc.env_model import AdamModel, QuadrotorModel
from safe_mpc.utils import  get_controller , get_ocp_acados, randomize_model, quat_to_rpy
from safe_mpc.robot_visualizer import DroneVisualizer
import copy
from safe_mpc.cost_definition import *

from acados_template import AcadosOcp, AcadosOcpSolver
import re


class QuadrotorController:
    def __init__(self, model):
        self.model = model
        self.current_step = 0

        self.N = self.model.params.N
        self.ocp = AcadosOcp()
        self.build_flag = False

        # Dimensions
        self.ocp.solver_options.tf = self.model.params.dt * self.N
        self.ocp.dims.N = self.N

        # Model
        self.ocp.model = self.model.amodel

        

        # Cost
        self.cost = None

        lbu = [0] * self.model.nu
        ubu = self.model.u_max

        INF = 1e5

        self.ocp.constraints.lbu = np.array(lbu)
        self.ocp.constraints.ubu = np.array(ubu)
        self.ocp.constraints.idxbu = np.array([0, 1, 2, 3])

        self.ocp.constraints.lbx_0 = -np.ones(self.model.nx) * INF
        self.ocp.constraints.ubx_0 = np.ones(self.model.nx) * INF
        self.ocp.constraints.idxbx_0 = np.arange(self.model.nx)

        self.ocp.constraints.lbx = np.array([-0.05, -0.05, -1.3])
        self.ocp.constraints.ubx = np.array([INF, INF, INF])
        self.ocp.constraints.idxbx = np.array([0, 1, 2])

        # dyn_constr_eqn = []

        # ineq_constr_eqn = []
        # ineq_constr_eqn = cs.vertcat(ineq_constr_eqn, dyn_constr_eqn)

        # self.model.con_h_expr = ineq_constr_eqn
        # self.model.con_h_expr_e = ineq_constr_eqn

        # # inequality bounds
        # nh = self.model.con_h_expr.shape[0]

        # lh = np.zeros(nh);        uh = np.zeros(nh)
        # lh[:] = -1e5;             uh[:] = 1e5

        # self.ocp.constraints.lh = lh
        # self.ocp.constraints.uh = uh
        # self.ocp.constraints.lh_e = lh
        # self.ocp.constraints.uh_e = uh

        self.ocp.solver_options.integrator_type = "ERK"
        self.ocp.solver_options.nlp_solver_type = "SQP"
        self.ocp.solver_options.sim_method_num_stages = 4
        self.ocp.solver_options.sim_method_num_steps = 1
        self.ocp.solver_options.globalization = 'MERIT_BACKTRACKING'
        self.ocp.solver_options.nlp_solver_max_iter = 300
        self.ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
        
        self.reset_controller()        

        # Reference and fails counter
        self.fails = 0

        # Empty initial guess and temp vectors
        self.x_guess = np.zeros((self.N + 1, self.model.nx))
        self.u_guess = np.zeros((self.N, self.model.nu))
        self.x_temp, self.u_temp = np.copy(self.x_guess), np.copy(self.u_guess)

        # Time stats
        self.time_fields = ['time_lin', 'time_sim', 'time_qp', 'time_qp_solver_call',
                            'time_glob', 'time_reg', 'time_tot']
        self.last_status = 4

    def checkSafeConstraints(self, x):
        return self.safe_set.check_constraint(x) 
    
    def create_safe_set(self):
        pass
    
    def additionalSetting(self):
        pass

    def solve(self, x0):
        if not(self.build_controller):
            raise ValueError("Controller not builded")
        
        # Reset current iterate
        self.ocp_solver.reset()

        # Constrain initial state
        self.ocp_solver.set(0, "lbx", x0)
        self.ocp_solver.set(0, "ubx", x0)

        target = np.array([0.6, 0.3, 0.2])

        yref = np.zeros(self.model.ny)
        for i in range(self.N):
            self.ocp_solver.set(i, 'x', self.x_guess[i])
            self.ocp_solver.set(i, 'u', self.u_guess[i])
            yref[:3] = target
            self.ocp_solver.set(i, 'yref', yref)
            
        self.ocp_solver.set(self.N, 'x', self.x_guess[-1])
        yref = np.zeros(self.model.nx)
        yref[:3] = target
        self.ocp_solver.set(self.N, 'yref', yref)

        status = self.ocp_solver.solve()

        # Save the temporary solution, independently of the status
        for i in range(self.N):
            self.x_temp[i] = self.ocp_solver.get(i, "x")
            self.u_temp[i] = self.ocp_solver.get(i, "u")
        self.x_temp[-1] = self.ocp_solver.get(self.N, "x")

        self.last_status = status
        return status
    
    def solve_and_sim(self):
        '''Solve the OCP with multiple shooting, and forward simulate with RK4'''

        # Integrate ODE model to get CL estimate (no measurement noise)
        u_0 = self.ocp_solver.solve_for_x0(self.zeta_0)
        self.zeta_0 = self.model.acados_integrator.simulate(x=self.zeta_0, u=u_0)

        u_0 = self.ocp_solver.solve_for_x0(self.zeta_0)
        self.zeta_N = np.reshape(self.ocp_solver.get(0, "x"), (self.model.nx, 1))
        for i in range(1, self.N +1):
            zeta_i = np.reshape(self.ocp_solver.get(i, "x"), (self.model.nx, 1))
            self.zeta_N = np.concatenate((self.zeta_N, zeta_i), axis = 1)

        self.u_N[:, 0] = u_0
        return 0

    def provideControl(self):
        """ Save the guess for the next MPC step and return u_opt[0] """
        if self.fails > 0:
            u = self.u_guess[0]
            # Rollback the previous guess
            self.x_guess = np.roll(self.x_guess, -1, axis=0)
            self.u_guess = np.roll(self.u_guess, -1, axis=0)
        else:
            u = self.u_temp[0]
            # Save the current temporary solution
            self.x_guess = np.roll(self.x_temp, -1, axis=0)
            self.u_guess = np.roll(self.u_temp, -1, axis=0)
        # Copy the last values
        self.x_guess[-1] = np.copy(self.x_guess[-2])
        self.u_guess[-1] = np.copy(self.u_guess[-2])
        return u, False

    def step(self, x0):
        pass

    def setReference(self, ee_ref):
        self.model.ee_ref = ee_ref

    def getTime(self):
        return np.array([self.ocp_solver.get_stats(field) for field in self.time_fields])

    def setGuess(self, x_guess, u_guess):
        self.x_guess = x_guess
        self.u_guess = u_guess

    def getGuess(self):
        return np.copy(self.x_guess), np.copy(self.u_guess)

    def getLastViableState(self):
        return np.copy(self.x_viable)
    
    def resetHorizon(self, N):
        self.N = N
        self.model.params.N = N
        self.ocp_solver.set_new_time_steps(np.full(N, self.model.params.dt))
        self.ocp_solver.update_qp_solver_cond_N(N)

        self.x_temp = np.zeros((self.N + 1, self.model.nx))
        self.u_temp = np.zeros((self.N, self.model.nu))

        self.cost.update_trajectory()

    def safeGuess(self, x, u, n_safe):
        """
        Function not more used. Its purpose was seeing if the viable state can be actually reached
        """
        for i in range(n_safe):
            x, _ = self.model.integrate(x, u[i])
            if not self.model.checkStateConstraints(x) or not self.checkCollision(x):
                return False, None
        return True, x
    
    def guessCorrection(self):
        """
        Correct guess, integrating with the plant dynamics known by the controller
        """
        for i in range(self.N):
            self.x_guess[i + 1] = self.model.integrate_naively(self.x_guess[i], self.u_guess[i])
        
    
    def reset_controller(self):
        self.fails = 0
        self.model.params.use_net = None
        self.net_name = ''
        self.current_step = 0

    def set_cost(self,cost):
        cost.set_solver_cost(self)
    
    def build_controller(self,build=True,name=''):
        self.ocp_name = "".join(re.findall('[A-Z][^A-Z]*', self.__class__.__name__)[:-1]).lower() + self.model.params.solver_type + self.net_name
        gen_name = self.model.params.GEN_DIR + 'ocp_' + self.ocp_name + '_' + self.model.amodel.name + '_' + name
        self.ocp.code_export_directory = gen_name
        self.ocp_solver = AcadosOcpSolver(self.ocp, json_file=gen_name + '.json', generate=build, build=build)
        self.build_flag = True

    def checkGuess(self):
        return self.model.checkRunningConstraints(self.x_temp, self.u_temp) and \
               self.model.checkDynamicsConstraints(self.x_temp, self.u_temp)


args = parse_args()
model_name = args['system']
params = Parameters(args, model_name, rti=False)
params.q_margin = args['joint_bounds_margin']
params.collision_margin = args['collision_margin']
params.build = args['build']
params.solver_type = 'SQP'
params.act = args['activation']
params.alpha = args['alpha']
params.N=args['horizon']

params.noise_mass = args['noise']
params.noise_inertia = args['noise']
params.noise_cm = args['noise']

model = QuadrotorModel(params)

cost = ReachTargetLS(model, params.Q_weight, params.R_weight, params.Qn_weight)
ocp_name = args['controller']
params.cont_name = args['controller']

ocp = QuadrotorController(model)

ocp.set_cost(cost)
ocp.build_controller(build = args['build'])
ocp.resetHorizon(args['horizon'])

num_ics = params.test_num
succ, fails, skip_ics = 0, 0, 0

#### SAMPLE ONLY THE INITIAL POSITION
sampler = qmc.Halton(3, scramble=False)

x_guess_naive, u_guess_naive = [], []
x_guess_zerovel, u_guess_zerovel = [], []

box_lb = np.array([-1., -1., -1.])
box_ub = np.array([1., 1., 1.])

progress_bar = tqdm(total=num_ics, desc=f'Generating initial conditions, alpha {ocp.model.params.alpha}')
start_time = time.time()

while succ < num_ics:

    q0 = qmc.scale(sampler.random(), box_lb, box_ub)[0]
    
    x0 = np.zeros((model.nx,))
    x0[3] = 1.          # Quaternion
    x0[:3] = q0

    u0_g = np.array([np.ones((model.nu,)) * params.mass * 9.81 / 4]*ocp.N)
    x0_g = np.array([x0]*(args['horizon']+1))
    ocp.setGuess(x0_g, u0_g)

    status = ocp.solve(x0)
    tqdm.write(f'Status: {status} , Check guess: {ocp.checkGuess()}')
    if (status == 0 or status == 2) and ocp.checkGuess():
        tqdm.write(f'Solver status {status}, x0 {x0}')
        xg_naive = copy.copy(ocp.x_temp)
        ug_naive = copy.copy(ocp.u_temp)
        x_guess_naive.append(xg_naive), u_guess_naive.append(ug_naive)
        succ += 1
        progress_bar.update(1)
        tqdm.write('SUCCESS')
    else:
        fails +=1
        
x_guess = copy.copy(ocp.x_temp)

print(f'Number of failed initializations: {fails}')
# u_guess = copy.copy(ocp.u_temp)
# print('Solution: \n', u_guess)

# Visualization
viz = DroneVisualizer(params)
target = np.array([0.6, 0.3, 0.2])

viz.setTarget(target)


time.sleep(4)

for i in range(num_ics):
    xg = x_guess_naive[i]
    ug = u_guess_naive[i]

    viz.display(xg[0])
    time.sleep(1)

    for j in range(params.N + 1):
        viz.display(xg[j])

    time.sleep(1)
