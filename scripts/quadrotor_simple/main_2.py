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

import numpy as np
import casadi as ca
import os
from common import *
from acados_settings import AcadosCustomOcp
import matplotlib.pyplot as plt

from safe_mpc.parser import Parameters, parse_args
from safe_mpc.env_model import AdamModel, QuadrotorModel

from safe_mpc.cost_definition import *

def plan_ocp( ocp_wrapper):
    '''Motion control problem of drone trajectory tracking'''

    # dimensions
    nx = ocp_wrapper.nx
    nu = ocp_wrapper.nu

    # initialize iteration variables
    t0 = 0
    discr_step = np.array([])
    times = np.array([[0]])
    mpc_iter = 0
    cost = 0
    fail = 0

    # Define states, controls, and slacks as coloumn vectors
    zeta_0 = np.copy(ocp_wrapper.zeta_0)
    u_0 = np.copy(U_REF)

    # Convert 1D array (nx,) to 3D (nx, 1, N+1)
    zeta_N = ca.repmat(np.reshape(zeta_0, (nx,1)), 1, N+1)

    # nx x N+1 x mpc_iter (to plot state during entire Horizon)
    state_steps = ca.repmat(zeta_0, 1, N+1)
    # nu x mpc_iter
    control_steps = ca.repmat(u_0, 1, 1)
    # 2 x mpc_iter (iteration time, cost)
    misc_step = ca.repmat(np.array([0, 0]).T, 1)

    # Control loop entry point

    for i in range(Nsim):
        # Store previous iterate data for plots
        discr_step = ca.reshape(np.array([t0, cost]), 2, 1)
        misc_step = np.concatenate( (misc_step, discr_step), axis = 1)
        state_steps = np.dstack((state_steps, ocp_wrapper.x_guess))
        control_steps = np.concatenate((control_steps,
                                        np.reshape(ocp_wrapper.u_guess[:, 0], (nu, 1))),
                                        axis = 1)
        t1 = time.time()

        target = np.array([0.6, 0.3, 0.2])
        end = ocp_wrapper.cost_update_ref(target)
        if (end) :
            print("Track complete !")
            break

        # Solve the OCP with updated state and controls
        ocp_wrapper.solve_and_sim(model)
        # ocp_wrapper.solve()
        ocp_wrapper.solver.print_statistics()

        # cost = ocp_wrapper.get_cost()

        t0 = round(t0 + T_del, 3)
        t2 = time.time()
        ocp_soln_time = t2-t1
        times = np.vstack(( times, ocp_soln_time))
        mpc_iter = mpc_iter + 1

        zeta_N = ocp_wrapper.x_guess

        print(f'\n Soln. {mpc_iter} Sim: {(np.round(ocp_wrapper.x_guess[:, 0],2).T)} at {round(t0,2)} s\t')

    sqp_max_sec = round(np.array(times).max(), 3)
    sqp_avg_sec = round(np.array(times).mean(), 3)
    avg_n = round(np.abs(state_steps[1, 0, :]).mean(), 4)
    avg_b = round(np.abs(state_steps[2, 0, :]).mean(), 4)

    print(f'Max. solver time\t\t: {sqp_max_sec * 1000} ms')
    print(f'Avg. solver time\t\t: {sqp_avg_sec * 1000} ms')
    print(f'Avg. lateral deviation n\t: {avg_n} m')
    print(f'Avg. vertical deviation b\t: {avg_b} m')


    return misc_step, state_steps, control_steps, target

if __name__ == '__main__':

    args = parse_args()
    model_name = args['system']
    params = Parameters(args, model_name, rti=False)
    params.q_margin = args['joint_bounds_margin']
    params.collision_margin = args['collision_margin']
    params.build = args['build']
    params.solver_type = 'SQP_RTI'
    params.act = args['activation']
    params.alpha = args['alpha']
    params.N=args['horizon']

    params.noise_mass = args['noise']
    params.noise_inertia = args['noise']
    params.noise_cm = args['noise']

    model = QuadrotorModel(params)

    custom_ocp = AcadosCustomOcp()
    custom_ocp.my_ocp_setup(params, model)
    traj_sample, traj_ST, traj_U, goal = plan_ocp(custom_ocp)

    viz = DroneVisualizer()
    viz.setTarget(goal)
    
    n_steps = traj_ST.shape[2]
    viz.display(traj_ST[0, 0, :3], traj_ST[0, 0, 3:7], T_del)

    time.sleep(4)

    rpy = np.zeros((n_steps, 3))
    for i in range(n_steps):
        viz.display(traj_ST[:3, 0, i], traj_ST[3:7, 0, i], T_del)
        rpy[i] = quat_to_rpy(traj_ST[3:7, 0, i])

    PLOT = 1

    if PLOT: 
        # Visualization
        t = np.arange(0, n_steps) * T_del
        fig, ax = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
        ax[0].plot(t, traj_ST[0, 0, :], c='r')
        ax[0].set_ylabel('x')
        ax[1].plot(t, traj_ST[1, 0, :], c='g')
        ax[1].set_ylabel('y')
        ax[2].plot(t, traj_ST[2, 0, :], c='b')
        ax[2].set_ylabel('z')
        ax[-1].set_xlabel('Time (s)')

        fig, ax = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
        ax[0].plot(t, rpy[:, 0], c='r')
        ax[0].set_ylabel('roll')
        ax[1].plot(t, rpy[:, 1], c='g')
        ax[1].set_ylabel('pitch')
        ax[2].plot(t, rpy[:, 2], c='b')
        ax[2].set_ylabel('yaw')
        ax[-1].set_xlabel('Time (s)')

        plt.figure()
        ax = plt.axes(projection='3d')
        ax.plot(traj_ST[0,0, :], traj_ST[1,0, :], traj_ST[2,0, :])
        # ax.plot(xref, yref, zref)
        ax.scatter(goal[0], goal[1], goal[2], c=[1,0,0], label='goal')
        ax.axis('equal')
        ax.set_xlabel('x [m]')
        ax.set_ylabel('y [m]')
        ax.set_zlabel('z [m]')
        ax.legend()

        t = np.arange(0, N+1) * T_del
        fig, ax = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
        for i in range(4):
            ax[0].plot(t, traj_ST[0, :, i], label='x_' + str(i))
            ax[0].set_ylabel('x')
            ax[1].plot(t, traj_ST[1, :, i], label='y_' + str(i))
            ax[1].set_ylabel('y')
            ax[2].plot(t, traj_ST[2, :, i], label='z_' + str(i))
            ax[2].set_ylabel('z')
            ax[0].legend()
            ax[-1].set_xlabel('Time (s)')
            t += T_del

        plt.show()