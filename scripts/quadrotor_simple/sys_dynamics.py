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
from common import *

'''Global Symbolic variables'''
# State variables

# Positions 
x = ca.MX.sym('x')
y = ca.MX.sym('y')
z = ca.MX.sym('z')

# # Quarternion heading (body frame )
q1 = ca.MX.sym('q1')
q2 = ca.MX.sym('q2')
q3 = ca.MX.sym('q3')
q4 = ca.MX.sym('q4')

# Transaltional velocities (inertial frame, m/s)
vx = ca.MX.sym('vx')
vy = ca.MX.sym('vy')
vz = ca.MX.sym('vz')
v_c = ca.vertcat(vx, vy, vz)

# Angular velocities w.r.t phi(roll), theta(pitch), psi(yaw)
# (body frame, m/s)
wr = ca.MX.sym('wr')
wp = ca.MX.sym('wp')
wy = ca.MX.sym('wy')
omg = ca.vertcat(wr, wp, wy)

# Control variable angles (Motor RPM)
ohm1 = ca.MX.sym('ohm1')
ohm2 = ca.MX.sym('ohm2')
ohm3 = ca.MX.sym('ohm3')
ohm4 = ca.MX.sym('ohm4')

zeta_f = ca.vertcat(x, y, z, q1, q2, q3, q4, wr, wp, wy, vx, vy, vz) #, ohm1, ohm2, ohm3, ohm4)

alpha1 = ca.MX.sym('alpha1')
alpha2 = ca.MX.sym('alpha2')
alpha3 = ca.MX.sym('alpha3')
alpha4 = ca.MX.sym('alpha4')
u = ca.vertcat( ohm1, ohm2, ohm3, ohm4)


class SysDyn():

    def __init__(self):

        self.n_samples = 0
        self.solver = None

    def SetupOde(self):
        '''ODEs for system dynamic model'''

        D = (Cd / mq) *ca.vertcat(vx*2, vy*2, vz**2)
        F = Ct * ca.vertcat(0, 0, ohm1**2  + ohm2**2  + ohm3**2  + ohm4**2 )
        G = ca.vertcat(0, 0, g0)
        J = np.diag([Ix, Iy, Iz])
        M = ca.vertcat(Ct * l * (ohm1**2 + ohm2**2 - ohm3**2 - ohm4**2),
                      Ct * l * (ohm1**2 - ohm2**2 - ohm3**2 + ohm4**2),
                      Cd * (ohm1**2 - ohm2**2 + ohm3**2 - ohm4**2))

        Rq = ca.vertcat(ca.horzcat( 2 * (q1**2 + q2**2) - 1,    -2 * (q1*q4 - q2*q3),       2 * (q1*q3 + q2*q4)),
                        ca.horzcat( 2 * (q1*q4 + q2*q3),         2 * (q1**2 + q3**2) - 1,   2 * (q1*q2 - q3*q4)),
                        ca.horzcat( 2 * (q1*q3 - q2*q4),         2 * (q1*q2 + q3*q4),       2 * (q1**2 + q4**2) - 1))

        # Orientation ODEs ( qauternion)
        q1Dot = (-(q2 * wr) - (q3 * wp) - (q4 * wy))/2
        q2Dot = ( (q1 * wr) - (q4 * wp) + (q3 * wy))/2
        q3Dot = ( (q4 * wr) + (q1 * wp) - (q2 * wy))/2
        q4Dot = (-(q3 * wr) + (q2 * wp) + (q1 * wy))/2

        # Cartesian velocity ODEs ( including drag)
        vDot_c = -G + (1/ mq) * Rq @ F - D

        # Angular velocity ODEs (rate of change of projected Euler angles)
        omgDot = ca.inv(J) @ (M - ca.cross(omg, J @ omg))

        # ohm1Dot = alpha1
        # ohm2Dot = alpha2
        # ohm3Dot = alpha3
        # ohm4Dot = alpha4

        
        dyn_f = ca.vertcat(vx, vy, vz, 
                           q1Dot, q2Dot, q3Dot, q4Dot,
                           omgDot[0], omgDot[1], omgDot[2],
                           vDot_c[0], vDot_c[1] , vDot_c[2], )
                        #    ohm1Dot, ohm2Dot, ohm3Dot, ohm4Dot)

        dyn_fun = ca.Function('f', [zeta_f, u], [dyn_f])

        return zeta_f, dyn_f, u, dyn_fun
