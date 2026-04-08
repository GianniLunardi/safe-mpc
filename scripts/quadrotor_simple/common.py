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
from pathlib import Path
from typing import Union
import meshcat
import time
import pinocchio as pin
from robot_descriptions.loaders.pinocchio import load_robot_description

'''Global variables'''

track="trefoil_track.txt"


# CrazyFlie 2.1 physical parameters
g0  = 9.80665       # [m.s^2] gravitational accerelation
mq  = 31e-3         # [kg] total mass (with Lighthouse deck)
Ix = 1.395e-5       # [kg.m^2] Inertial moment around x-axis
Iy = 1.395e-5       # [kg.m^2] Inertial moment around y-axis
Iz = 2.173e-5       # [kg.m^2] Inertia moment around z-axis
Cd  = 7.9379e-06    # [N/krpm^2] Drag coefficient
Ct  = 3.25e-4       # [N/krpm^2] Thrust coefficient
dq  = 92e-3         # [m] distance between motors' center
l   = dq/2          # [m] distance between motors' center and the axis of rotation

INF = 1e5

# timing parameters
T_del = 0.005               # time between steps in seconds
N = 100                     # number of shooting nodes
Tf = N * T_del

Tsim = 45
Nsim = int(Tsim * N / Tf)

U_MAX = 22                                              # [krpm]
U_HOV = int(np.sqrt(.25 * 1e6* mq * g0 /Ct)) /1000      #[krpm]
U_REF = np.array([U_HOV, U_HOV, U_HOV, U_HOV])

# State
n_states = 17

init_zeta = np.array([0.05, 0, 0,       # x,  y,  z
                      1, 0, 0, 0,       # qw, qx, qy, qz,
                      0, 0, 0,          # ohmr,  ohmp,  ohmy
                      0, 0, 0,  ])        # vx, vy, vz
                    #   U_HOV, U_HOV, U_HOV, U_HOV ])     # ohm1, ohm2, ohm3, ohm4

rob_rad = 0.04                           # radius of the drone covering sphere

# Control
n_controls = 4

# Weights & Tracking reference
S_REF = 0.1875
S_MAX = 5.9
                                          # State weights on
Q = np.diag([100, 100, 100,                     # xyz position
             1e-3, 1e-3, 1e-3, 1e-3,      # quaternion
             1e-3, 1e-3, 1e-3,            # drone angular velocity
             1e-3, 1e-3, 1e-3, ])           # cartesian velocity
            #  1e-8, 1e-8, 1e-8, 1e-8])     # rotor angular velocity

                                          # Terminal state weights on
Qn = np.diag([100, 100, 100,                 # xyz position
             1e-5, 1e-5, 1e-5, 1e-5,      # quaternion
             1e-5, 1e-5, 1e-5,            # drone angular velocity
             1e-5, 1e-5, 1e-5, ])           # cartesian velocity
            #  1e-8, 1e-8, 1e-8, 1e-8])     # rotor angular velocity

R = np.diag([1e-5, 1e-5, 1e-5, 1e-5])

#  MatPlotLib animation settings
Tstart_offset = 0
f_plot = 10
refresh_ms = 10
sphere_scale = 20000 #TODO make dependant on map size. (10000/ 20 obst)
z_const = 0.1

''' Helper functions'''

def DM2Arr(dm):
    return np.array(dm.full(), dtype=object)

def quat2rpy(qoid):
    ''' qoid -> [qw, qx, qy, qz]
        returns euler angles in degrees
        reference math3d.h crazyflie-firmware'''

    r	  =  ca.atan2( 2 * (qoid[0]*qoid[1] + qoid[2]*qoid[3]), 1 - 2 * (qoid[1]**2 + qoid[2]**2 ))
    p     =  ca.asin( 2 *  (qoid[0]*qoid[2] - qoid[1]*qoid[3]))
    y	  =  ca.atan2( 2 * (qoid[0]*qoid[3] + qoid[1]*qoid[2]), 1 - 2 * (qoid[2]**2 + qoid[3]**2 ))

    r_d = r * 180 / np.pi          # roll in degrees
    p_d = p * 180 / np.pi          # pitch in degrees
    y_d = y * 180 / np.pi          # yaw in degrees

    return [r_d, p_d, y_d]

def get_norm_2(diff):

    norm = ca.sqrt(diff.T @ diff)

    return norm

def get_2norm_2(diff):

    norm = (diff.T @ diff)

    return norm

def get_2norm_W(diff, W):

    norm = (diff.T @ W @diff)

    return norm


def get_norm_W(diff, W):

    norm = ca.sqrt(diff.T @ W @diff)

    return norm

def quat_to_rot(q):
    w, x, y, z= q

    R = np.array([
        [1 - 2*(y*y + z*z),     2*(x*y - z*w),     2*(x*z + y*w)],
        [2*(x*y + z*w),     1 - 2*(x*x + z*z),     2*(y*z - x*w)],
        [2*(x*z - y*w),         2*(y*z + x*w), 1 - 2*(x*x + y*y)]
    ])
    return R

def quat_to_rpy(q):
    w, x, y, z= q

    # Roll (x-axis rotation)
    sinr_cosp = 2 * (w*x + y*z)
    cosr_cosp = 1 - 2 * (x*x + y*y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    # Pitch (y-axis rotation)
    sinp = 2 * (w*y - z*x)
    sinp = np.clip(sinp, -1.0, 1.0)  # evita errori numerici
    pitch = np.arcsin(sinp)

    # Yaw (z-axis rotation)
    siny_cosp = 2 * (w*z + x*y)
    cosy_cosp = 1 - 2 * (y*y + z*z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)

    return np.array([roll, pitch, yaw])

class DroneVisualizer:
    def __init__(self):
        drone = load_robot_description("cf2_description")
        
        self.viz = pin.visualize.MeshcatVisualizer(drone.model, drone.collision_model, drone.visual_model)
        self.viz.initViewer(loadModel=True, open=True)

        # Set the end-effector target
        ee_radius = 0.075  
        sphere = meshcat.geometry.Sphere(ee_radius)
        self.viz.viewer['world/robot/target'].set_object(sphere)
        self.viz.viewer['world/robot/target'].set_property('color', [0, 1, 0, 0.4])
        self.viz.viewer['world/robot/target'].set_property('visible', False)
        T_target = np.eye(4)
        self.viz.viewer['world/robot/target'].set_transform(T_target)

    def display(self, pos, quat, dt):
        pose = np.eye(4)
        pose[:3, 3] = pos
        pose[:3, :3] = quat_to_rot(quat)
        self.viz.viewer['pinocchio/visuals/base_link_0'].set_transform(pose)
        time.sleep(dt)
    
    def setTarget(self, ee_ref):
        T_target = np.eye(4)
        T_target[:3, 3] = ee_ref
        self.viz.viewer['world/robot/target'].set_transform(T_target)
        self.viz.viewer['world/robot/target'].set_property('visible', True)