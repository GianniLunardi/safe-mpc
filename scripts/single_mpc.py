import pickle
import numpy as np
from functools import reduce
from safe_mpc.parser import Parameters, parse_args
from safe_mpc.env_model import AdamModel
from safe_mpc.utils import get_controller, randomize_model
from safe_mpc.controller import SafeBackupController
from safe_mpc.cost_definition import *
import matplotlib.pyplot as plt


CALLBACK = True

args = parse_args()
model_name = args['system']
params = Parameters(model_name, rti=True)
params.build = args['build']
params.act = args['activation']
params.alpha = args['alpha']
params.backhor = args['back_hor']
horizon = args['horizon']
params.N=horizon
model = AdamModel(params)
model.ee_ref = params.ee_ref
nq = model.nq

params.noise_mass = args['noise']
params.noise_inertia = args['noise']
params.noise_cm = args['noise']

build_controllers=args['build']

cont_name = args['controller']
controller = get_controller(cont_name, model)
# cost_controller = TrackingMovingCircleEXT(model,params.Q_weight,params.R_weight)
cost_controller = ReachTargetEXT(model,params.Q_weight,params.R_weight)
cost_controller.set_solver_cost(controller)
controller.build_controller(build_controllers)

param_backup = Parameters(model_name, rti=True)
param_backup.use_net = None
param_backup.N = args['back_hor']
param_backup.solver_type = 'SQP_RTI'
model_backup = AdamModel(param_backup)
safe_ocp = SafeBackupController(model_backup)
cost_controller_backup = ZeroCost(model_backup)
cost_controller_backup.set_solver_cost(safe_ocp)
safe_ocp.build_controller(build=args['build'])

controller.resetHorizon(horizon)
safe_ocp.resetHorizon(params.back_hor)

traj__track = 'traj_track' if controller.model.params.track_traj else "" 

data = pickle.load(open(f'{params.DATA_DIR}{model_name}_{cont_name}_{horizon}hor_{int(params.alpha)}sm_use_net{controller.model.params.use_net}_{traj__track}_guess.pkl', 'rb'))
print(f'{params.DATA_DIR}{model_name}_{cont_name}_{horizon}hor_{int(params.alpha)}sm_use_net{controller.model.params.use_net}_{traj__track}_guess.pkl')

x_guess = data['xg']
u_guess = data['ug']
x_init = x_guess[:,0,:]


# MPC simulation 
traj_costs = [[] for _ in range(x_init.shape[0])]
EVAL = False

counters = np.zeros(5)
not_conv = 0
tau_viol = []
kp, kd = 0.1, 1e2
print(x_init.shape[0])

t = np.arange(0, params.n_steps) * params.dt

i = 37          # 42

if controller.model.params.track_traj:
    randomize_model(params.robot_urdf, noise_mass = params.noise_mass, noise_inertia = params.noise_inertia, noise_cm_position = params.noise_cm)
    controller.model.update_randomized_dynamics()
    safe_ocp.model.update_randomized_dynamics()

traj_costs[i] = 0
print(f'Simulation {i + 1}/{params.test_num}')
x0 = x_init[i]
x_sim = np.empty((params.n_steps + 1, model.nx)) * np.nan
u = np.empty((params.n_steps, model.nu)) * np.nan
r_index = np.empty((params.n_steps, 1)) * np.nan 
x_sim[0] = x0

nn_out_r = np.empty(params.n_steps) * np.nan
nn_out_t = np.empty(params.n_steps) * np.nan

controller.setGuess(x_guess[i], u_guess[i])
controller.reset_controller()

j = 0
ja = 0
sa_flag = False
for j in range(params.n_steps):
    # if controller.track_traj:
    #     traj_costs[i] += (controller.model.jointToEE(x_sim[-1])-controller.traj_to_track[:,1]).T @ controller.Q @ (controller.model.jointToEE(x_sim[-1])-controller.traj_to_track[:,1])
    if sa_flag and ja < safe_ocp.N:
        # Follow safe abort trajectory (PD to stabilize at the end)
        u[j] = u_abort[ja]
        ja += 1
        
    else:   
        u[j], sa_flag = controller.step(x_sim[j])
        if cont_name in ['receding', 'parallel']:
            r_index[j] = controller.r
            nn_out_r[j] = controller.safe_set.nn_func_x(controller.x_temp[controller.r])
            nn_out_t[j] = controller.safe_set.nn_func_x(controller.x_temp[-1])

        # Check Safe Abort
        if sa_flag:
            x_viable = controller.getLastViableState()
            if CALLBACK:
                if params.urdf_name == 'z1':
                    print(f'  ABORT at step {j}, x = {x_viable}')
                    if controller.model.params.use_net:
                        print(f'  NN output at abort with current alpha {int(params.alpha)}: ' 
                            f'{controller.safe_set.nn_func_x(x_viable)}')
                        print(f'  NN output at abort with alpha = 10: '
                            f'{controller.safe_set.nn_func(x_viable, [0,0,0,10.,0])}')
            # Instead of breaking, solve safe abort problem
            xg = np.full((safe_ocp.N + 1, model.nx), x_viable)
            ug = np.zeros((safe_ocp.N, model.nu))
            safe_ocp.setGuess(xg, ug)
            status = safe_ocp.solve(x_viable)
            print(f' Number of SQP iterations: {controller.ocp_solver.get_stats("sqp_iter")}')

            if status != 0:
                if CALLBACK:
                    print('  SAFE ABORT FAILED')
                    print('  Current controller fails:', controller.fails)
                break
            ja = 0
            x_abort, u_abort = safe_ocp.x_temp, safe_ocp.u_temp

    tau = np.array([model.tau_fun(controller.x_temp[k], controller.u_temp[k]).T for k in range(len(controller.u_temp))])
    tau = np.array([model.tau_fun(controller.x_guess[k], controller.u_guess[k]).T for k in range(len(controller.u_guess))])
    
    if not model.checkStateConstraints(controller.x_temp):
        counters[0] += 1
        if EVAL:
            print(f'\tx Bounds violated at step {j}')
            for k in range(len(controller.x_temp)):
                viol = np.min(np.vstack((model.x_max - controller.x_temp[k], controller.x_temp[k] - model.x_min)), axis=0)
                if np.any(viol + params.tol_x < 0):
                    print(f'\t\tState {k} out of bounds: {viol}')
    if not model.checkTorqueBounds(tau):
        counters[1] += 1
        if EVAL:
            print(f'Step: {j}')

            print(f'\ttau Bounds violated at step {j}')
            for k in range(len(tau)):
                viol = model.tau_max - np.abs(tau[k])
                if np.any(viol + params.tol_tau < 0):
                    # Collect any violation, taking the minimum (so max violation)
                    tau_viol.append(np.min(viol))
                    print(f'\t\tTorque {k} out of bounds: {viol}')
            
    if not np.all([controller.model.checkCollision(x) for x in controller.x_temp]):
        counters[2] += 1
        if EVAL:
            print(f'\tCollision at step {j}')
            for k in range(len(controller.x_temp)):
                t_glob = model.jointToEE(controller.x_temp[k])
                for obs in controller.model.params.obstacles:
                    if obs['name'] == 'floor':
                        viol = t_glob[2] - obs['bounds'][0]
                        if viol + params.tol_obs < 0:
                            print(f'\t\tCollision {k} with floor: {viol}')
                    if obs['name'] == 'ball':
                        viol = np.sum((t_glob.flatten() - obs['position']) ** 2) - obs['bounds'][0]
                        if viol + params.tol_obs < 0:
                            print(f'\t\tCollision {k} with ball: {viol}')

    if cont_name not in ['naive', 'zerovel', 'trivial']:
        r = controller.r if (cont_name == 'receding' or cont_name == 'parallel') else -1
        if not controller.checkSafeConstraints(controller.x_temp[r]):
            counters[3] += 1
        if controller.last_status == 4:
            counters[4] += 1

    x_sim[j + 1], _ = model.integrate(x_sim[j], u[j])

    # Check next state bounds and collision
    if not model.checkStateBounds(x_sim[j + 1]):   
        if CALLBACK:
            print('  FAIL BOUNDS')
            print(f'\tState {j + 1} violation: {np.min(np.vstack((model.x_max - x_sim[j + 1], x_sim[j + 1] - model.x_min)), axis=0)}')
            print(f'\tCurrent controller fails: {controller.fails}')
        break
    if not controller.model.checkCollision(x_sim[j + 1]):
        if CALLBACK:
            print(f'  FAIL COLLISION at step {j + 1}')
        break
    # Check convergence
    if j == params.n_steps -1:
        # if i in viable_idx:
        #     viable_idx.remove(i)
        not_conv+=1
        if CALLBACK:
            print('not converged')

if np.linalg.norm(model.jointToEE(x_sim[-1]).T - model.ee_ref) < params.tol_conv:
    if CALLBACK:
        print('  SUCCESS !!')
print(f'initial state: {x_sim[0]}')

fix, ax = plt.subplots(3, 1, figsize=(16,9), sharex=True)
ax[0].plot(t, r_index, label='r', c='r')
ax[0].legend()
ax[1].plot(t, nn_out_r, label='NN on r', c='b')
ax[1].axhline(0, ls='--', c='black')
ax[1].legend()
ax[2].plot(t, nn_out_t, label='NN last', c='g')
ax[2].axhline(0, ls='--', c='black')
ax[2].legend()
ax[2].set_xlabel('Time (s)')

plt.show()
