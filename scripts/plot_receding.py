import numpy as np
import matplotlib.pyplot as plt
import pickle
from safe_mpc.parser import Parameters, parse_args
from safe_mpc.env_model import AdamModel
from safe_mpc.utils import get_controller

args = parse_args()
model_name = args['system']
cont_name = args['controller']
params = Parameters(model_name, rti=True)
hor = args['horizon']
alpha = int(args['alpha'])
model = AdamModel(params)
controller = get_controller(cont_name, model)
# traj__track = 'traj_track' if controller.model.params.track_traj else "" 

data = pickle.load(open(f'{params.DATA_DIR}z1_parallel_use_netTrue_45hor_10sm_mpc.pkl', 'rb'))

t = np.arange(0, params.n_steps) * params.dt

ic = 19
for ic in range(len(data['x'])):
    x, u, r = data['x'][ic], data['u'][ic], data['r'][ic]


    nn_out = np.empty(params.n_steps) * np.nan
    for i in range(params.n_steps):
        nn_out[i] = controller.safe_set.nn_func_x(x[i])

    # Receding index
    # plt.figure()
    # plt.plot(t, r, label='r')
    # plt.xlabel('Time (s)')
    # plt.ylabel('Receding index')
    # plt.grid(True)

    fig, ax = plt.subplots(2, 1, figsize=(10, 12), sharex=True)
    ax[0].plot(t, r, c='b')
    ax[0].set_ylabel('receding node')
    ax[1].plot(t, nn_out, c='r')
    ax[1].set_ylabel('NN out on current state')
    ax[1].set_xlabel('Time (s)')


    plt.savefig(f'rec_plot/r_{ic}.png')
    plt.close()