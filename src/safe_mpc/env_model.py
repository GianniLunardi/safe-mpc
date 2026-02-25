import re
import numpy as np
from copy import deepcopy
from urdf_parser_py.urdf import URDF
import adam
from adam.casadi import KinDynComputations
import casadi as cs
from casadi import MX, vertcat, Function
from acados_template import AcadosModel, AcadosSim, AcadosSimSolver
import scipy.linalg as lin
import torch.nn as nn
import l4casadi as l4c
from .safe_set import NetSafeSet, AnalyticSafeSet
import xml.etree.ElementTree as ET
from .utils import rot_mat_x,rot_mat_y,rot_mat_z, casadi_segment_dist,ball_segment_dist,sphere_sphere_dist,plane_sphere_dist, randomize_model


class AdamModel:
    def __init__(self, params):
        self.params = params
        self.amodel = AcadosModel()
        # Robot dynamics with Adam (IIT)                
        robot_joints = []
        jj=0
        for jointt in self.params.robot_descr.joints:
            if jointt.type != 'fixed':
                robot_joints.append(jointt)
                jj +=1
                if jj == self.params.nq:
                    break

        joint_names = [joint.name for joint in robot_joints]

        self.rng = np.random.default_rng(seed=0)


        #randomize_model(params.robot_urdf, noise_mass = 0, noise_inertia = 0, noise_cm_position = 0)

        # Formal assumed model, used by controller
        self.kin_dyn = KinDynComputations(params.robot_urdf, joint_names, self.params.robot_descr.get_root())        
        self.kin_dyn.set_frame_velocity_representation(adam.Representations.MIXED_REPRESENTATION)
        self.mass = self.kin_dyn.mass_matrix_fun()                           # Mass matrix
        self.bias = self.kin_dyn.bias_force_fun()                            # Nonlinear effects  
        self.gravity = self.kin_dyn.gravity_term_fun()                       # Gravity vector
        self.fk = self.kin_dyn.forward_kinematics_fun(params.frame_name)     # Forward kinematics

        # Model with noise
        self.kin_dyn_noisy = KinDynComputations(params.robot_urdf, joint_names, self.params.robot_descr.get_root())        
        self.kin_dyn_noisy.set_frame_velocity_representation(adam.Representations.MIXED_REPRESENTATION)
        self.mass_noisy = self.kin_dyn_noisy.mass_matrix_fun()                           # Mass matrix
        self.bias_noisy = self.kin_dyn_noisy.bias_force_fun()                            # Nonlinear effects  
        self.gravity_noisy = self.kin_dyn_noisy.gravity_term_fun()                       # Gravity vector
        self.fk_noisy = self.kin_dyn_noisy.forward_kinematics_fun(params.frame_name)     # Forward kinematics

        nq = len(joint_names)

        self.amodel.name = params.urdf_name
        self.x = MX.sym("x", nq * 2)
        self.x_dot = MX.sym("x_dot", nq * 2)
        self.u = MX.sym("u", nq)
        
        # Double integrator
        self.f_disc = vertcat(
            self.x[:nq] + params.dt * self.x[nq:] + 0.5 * params.dt**2 * self.u,
            self.x[nq:] + params.dt * self.u
        ) 
        self.f_fun = Function('f', [self.x, self.u], [self.f_disc])
            
        self.amodel.x = self.x
        self.amodel.u = self.u
        self.amodel.disc_dyn_expr = self.f_disc

        self.nx = self.amodel.x.size()[0]
        self.nu = self.amodel.u.size()[0]
        self.ny = self.nx + self.nu
        self.nq = self.params.nq
        self.nv = nq

        # Inverse dynamics, torque computation
        H_b = np.eye(4)
        self.tau = self.mass(H_b, self.x[:nq])[6:, 6:] @ self.u + \
                   self.bias(H_b, self.x[:nq], np.zeros(6), self.x[nq:])[6:]
        self.tau_fun = Function('tau', [self.x, self.u], [self.tau])

        # Noisy dynamics
        H_b = np.eye(4)
        self.tau_noisy = self.mass_noisy(H_b, self.x[:nq])[6:, 6:] @ self.u + \
                   self.bias_noisy(H_b, self.x[:nq], np.zeros(6), self.x[nq:])[6:]
        self.tau_noisy_fun = Function('tau', [self.x, self.u], [self.tau_noisy])

        # EE position (global frame)
        T_ee = self.fk(np.eye(4), self.x[:nq])
        self.t_loc = self.params.ee_pos
        self.t_glob = T_ee[:3, 3] + T_ee[:3, :3] @ self.t_loc
        self.ee_fun = Function('ee_fun', [self.x], [self.t_glob])

        # Noisy EE position (global frame)
        T_ee_noisy = self.fk_noisy(np.eye(4), self.x[:nq])
        self.t_loc_noisy = self.params.ee_pos
        self.t_glob_noisy = T_ee_noisy[:3, 3] + T_ee_noisy[:3, :3] @ self.t_loc_noisy
        self.ee_fun_noisy = Function('ee_fun', [self.x], [self.t_glob_noisy])

        # EE jacobian
        self.jac = self.kin_dyn.jacobian_fun(params.frame_name)

        # Joint limits
        joint_lower = np.array([joint.limit.lower for joint in robot_joints])
        joint_upper = np.array([joint.limit.upper for joint in robot_joints])
        joint_velocity = np.array([joint.limit.velocity for joint in robot_joints]) 
        joint_effort = np.array([joint.limit.effort for joint in robot_joints]) 
        joint_effort = joint_effort[:nq]

        self.tau_min = - joint_effort
        self.tau_max = joint_effort
        self.x_min = np.hstack([joint_lower, - joint_velocity])
        self.x_max = np.hstack([joint_upper, joint_velocity])

        self.u_min = -1e6 * np.ones(self.nu)
        self.u_max = 1e6 *np.ones(self.nu)

        # Bounds on control --> 0 since acceleration is not limited
        self.num_bound_u = 0

        self.bounds_diff = np.abs(self.x_max-self.x_min)

        self.x_min -= self.bounds_diff*(self.params.q_margin/100)
        self.x_max += self.bounds_diff*(self.params.q_margin/100)

        # EE target
        self.ee_ref = self.params.ee_ref

        # Cartesian constraints
        self.obs_string = self.params.obs_string
        self.joint_names = joint_names
        
        # Capsules end-points forward kinematics
        n_cap=0
        for capsule in self.params.robot_capsules:
            capsule['index']=n_cap
            rot_mat=np.eye(4)
            if capsule['rotation_offset'] != None:
                th_off=capsule['rotation_offset']
                rot_mat = rot_mat_x(th_off[0])@rot_mat_y(th_off[1])@rot_mat_z(th_off[2])
            if capsule['spatial_offset'] != None:
                prism_mat = np.array([[1,0,0,capsule['spatial_offset'][0]],
                                      [0,1,0,capsule['spatial_offset'][1]],
                                      [0,0,1,capsule['spatial_offset'][2]],
                                      [0,0,0,1]])
                rot_mat = prism_mat@rot_mat  
            fk_capsule_points = self.kin_dyn.forward_kinematics_fun(capsule['link_name'])   
            T_capsule_points = fk_capsule_points(np.eye(4), self.x[:self.nq])@rot_mat
            capsule['end_points_fk'] = deepcopy([(T_capsule_points @ capsule['end_points'][0])[:3],
                                                 (T_capsule_points @ capsule['end_points'][1])[:3]])
            capsule['end_points_T_fun'] = deepcopy(cs.Function(f'fun_T_{n_cap}',[self.x],[T_capsule_points]))
            capsule['end_points_fk_fun'] = deepcopy(cs.Function(f'fun_fk_{n_cap}',[self.x],[capsule['end_points_fk'][0][:3],
                                                                                            capsule['end_points_fk'][1][:3]]))
            n_cap += 1
        for capsule in self.params.obst_capsules:
            capsule['index']=n_cap
            capsule['end_points_fk_fun'] = deepcopy(cs.Function(f'fun_fk_{n_cap}',[self.x],[capsule['end_points'][0], capsule['end_points'][1]]))
            n_cap += 1
        n_cap = 0
        for sphere in self.params.spheres_robot:
            fk_sphere = self.kin_dyn.forward_kinematics_fun(sphere['link_name'])
            T_sphere = fk_sphere(np.eye(4), self.x[:nq])
            sphere['fk'] = T_sphere[:3,3] +T_sphere[:3, :3]@ sphere['spatial_offset']      
            sphere['fk_fun'] = Function(f'sphere_fk_{n_cap}', [self.x], [sphere['fk']])
            sphere['index']=n_cap
            n_cap +=1

        self.NL_external = self.generate_NLconstraints_list()

    def jointToEE(self, x):
        return np.array(self.ee_fun(x))

    def checkStateConstraints(self, x):
        return np.all(np.logical_and(x >= self.x_min - self.params.tol_x, 
                                     x <= self.x_max + self.params.tol_x)) and \
                                     self.checkCollision(x)

    def checkStateBounds(self, x):
        return np.all(np.logical_and(x >= self.x_min - self.params.tol_x, 
                                     x <= self.x_max + self.params.tol_x))

    def checkTorqueConstraints(self, x,u):
        tau = np.array([self.tau_fun(x[i], u[i]).T for i in range(len(u))])
        return np.all(np.logical_and(tau >= self.tau_min - self.params.tol_tau, 
                                     tau <= self.tau_max + self.params.tol_tau))
    
    def checkTorqueBounds(self,tau):
        return np.all(np.logical_and(tau >= self.tau_min - self.params.tol_tau, 
                                     tau <= self.tau_max + self.params.tol_tau))

    def checkRunningConstraints(self, x, u):
        return self.checkStateConstraints(x) and self.checkTorqueConstraints(x,u)

    
    def integrate(self, x, u):
        x_next = np.zeros(self.nx)
        tau = np.array(self.tau_noisy_fun(x, u).T)
        # Add Gaussian noise to the control
        tau += self.rng.normal(np.zeros(self.nu),self.tau_max*(self.params.control_noise/100),size=self.nu)
        # Cannot exceed the torque limits --> sat and compute forward dynamics on real system 
        H_b = np.eye(4)
        tau_sat = np.clip(tau, self.tau_min, self.tau_max)
        M = np.array(self.mass_noisy(H_b, x[:self.nq])[6:, 6:])
        h = np.array(self.bias_noisy(H_b, x[:self.nq], np.zeros(6), x[self.nq:])[6:])
        u = np.linalg.solve(M, (tau_sat.T - h)).T
        # x_next[:self.nq] = x[:self.nq] + self.params.dt * x[self.nq:] + 0.5 * self.params.dt**2 * u
        # x_next[self.nq:] = x[self.nq:] + self.params.dt * u
        x_next = np.array(self.f_fun(x,u)).squeeze()
        return x_next, u
    
    def integrate_naively(self, x, u):
        x_next = np.array(self.f_fun(x,u)).squeeze()
        return x_next

    def integrate_controller_model(self, x, u):
        x_next = np.zeros(self.nx)
        tau = np.array(self.tau_fun(x, u).T)
        if not self.checkTorqueBounds(tau):
            # Cannot exceed the torque limits --> sat and compute forward dynamics on real system 
            H_b = np.eye(4)
            tau_sat = np.clip(tau, self.tau_min, self.tau_max)
            M = np.array(self.mass(H_b, x[:self.nq])[6:, 6:])
            h = np.array(self.bias(H_b, x[:self.nq], np.zeros(6), x[self.nq:])[6:])
            u = np.linalg.solve(M, (tau_sat.T - h)).T
        x_next[:self.nq] = x[:self.nq] + self.params.dt * x[self.nq:] + 0.5 * self.params.dt**2 * u
        x_next[self.nq:] = x[self.nq:] + self.params.dt * u
        return x_next, u
    
    def checkDynamicsConstraints(self, x, u):
        # Rollout the control sequence
        n = np.shape(u)[0]
        x_sim = np.zeros((n + 1, self.nx))
        x_sim[0] = np.copy(x[0])
        for i in range(n):
            x_sim[i + 1], _ = self.integrate_controller_model(x_sim[i], u[i])
        # Check if the rollout state trajectory is almost equal to the optimal one
        return np.linalg.norm(x - x_sim) < self.params.tol_dyn * np.sqrt(n+1) 
    
    def checkCollision(self, x):
        x_tmp = np.atleast_2d(deepcopy(x))
        for i in range(len(x_tmp)):
            for pair in self.collisions_constr_fun:
                if not(pair[1]<=pair[0](x_tmp[i])<=pair[2]):
                    # print(f'collision: {pair[0].name()}')
                    return False
            return True

    
    def generate_NLconstraints_list(self):
        """
        Generate list of nonlinear constraints with bounds, for nodes 0, 1 - N-1, and N, as well as the list of casadi function of the collision constraints
        """
        constraint_list_0 = []
        constraint_list_1_N_minus_1 = []
        constraint_list_N = []

        # generate also list of collision function for collision checks
        self.collisions_constr_fun = []

        # torque limits constraints present
        constraint_list_0.append([self.tau,self.tau_min,self.tau_max])
        constraint_list_1_N_minus_1.append([self.tau,self.tau_min,self.tau_max])

        # collisions
        # if self.params.use_capsules:
        for i,pair in enumerate(self.params.collisions_pairs):
            if pair['type'] == 'capsule-capsule':
                constraint_list_0.append([casadi_segment_dist(*pair['elements'][0]['end_points_fk'],*pair['elements'][1]['end_points_fk']), \
                                        (pair['elements'][0]['radius']+pair['elements'][1]['radius']+self.params.collision_margin*2)**2,1e6])
                self.collisions_constr_fun.append([cs.Function(f"collision_constraint_{i}_{pair['elements'][0]['name']}_{pair['elements'][1]['name']}",[self.x],[constraint_list_0[-1][0]]), \
                                        (pair['elements'][0]['radius']+pair['elements'][1]['radius'])**2-self.params.tol_obs,1e6+self.params.tol_obs])
                # constraint_list_0[-1][1] += self.params.collision_margin**2
                constraint_list_1_N_minus_1.append(constraint_list_0[-1])
                constraint_list_N.append(constraint_list_0[-1])
                # print(pair['type'])
                # print(f'constraint {constraint_list_N[-1][1]}  constr_fun {self.collisions_constr_fun[-1][1]}')
            elif pair['type'] == 'capsule-sphere':
                constraint_list_0.append([ball_segment_dist(*pair['elements'][0]['end_points_fk'],pair['elements'][0]['length'],pair['elements'][1]['position']), \
                                        (pair['elements'][1]['radius']+pair['elements'][0]['radius']+self.params.collision_margin*2)**2,1e6])
                self.collisions_constr_fun.append([cs.Function(f"collision_constraint_{i}_{pair['elements'][0]['name']}_{pair['elements'][1]['name']}",[self.x],[constraint_list_0[-1][0]]), \
                                        (pair['elements'][1]['radius']+pair['elements'][0]['radius'])**2-self.params.tol_obs,1e6+self.params.tol_obs])
                # constraint_list_0[-1][1] += self.params.collision_margin**2
                constraint_list_1_N_minus_1.append(constraint_list_0[-1])
                constraint_list_N.append(constraint_list_0[-1])
                # print(pair['type'])
                # print(f'constraint {constraint_list_N[-1][1]}  constr_fun {self.collisions_constr_fun[-1][1]}')
                
            elif pair['type'] == 'capsule-plane':
                for point in pair['elements'][0]['end_points_fk']:
                    constraint_list_0.append([point[pair['elements'][1]['perpendicular_axis']],pair['elements'][1]['bounds'][0]+pair['elements'][0]['radius']+2*self.params.collision_margin,pair['elements'][1]['bounds'][1]-pair['elements'][0]['radius']-2*self.params.collision_margin])
                    self.collisions_constr_fun.append([cs.Function(f"collision_constraint_{i}_{pair['elements'][0]['name']}_{pair['elements'][1]['name']}",[self.x],[constraint_list_0[-1][0]]), \
                                        pair['elements'][1]['bounds'][0]+pair['elements'][0]['radius']-self.params.tol_obs,pair['elements'][1]['bounds'][1]-pair['elements'][0]['radius']+self.params.tol_obs])
                    # constraint_list_0[-1][1] += self.params.collision_margin
                    # constraint_list_0[-1][2] -= self.params.collision_margin
                    constraint_list_1_N_minus_1.append(constraint_list_0[-1])
                    constraint_list_N.append(constraint_list_0[-1])
                    # print(pair['type'])
                    # print(f'constraint {constraint_list_N[-1][1]}  constr_fun {self.collisions_constr_fun[-1][1]}')

                    
            elif pair['type'] == 'sphere-sphere':
                #constr_expr = sphere_sphere_dist(pair['elements'][1],pair['elements'][0]['fk'])
                constr_expr = (self.t_glob - pair['elements'][1]['position']).T @ (self.t_glob - pair['elements'][1]['position'])
                constraint_list_0.append([constr_expr, (pair['elements'][0]['radius']+pair['elements'][1]['radius'] + self.params.collision_margin*2)**2,1e6])
                self.collisions_constr_fun.append([cs.Function(f"collision_constraint_{i}_{pair['elements'][0]['name']}_{pair['elements'][1]['name']}",[self.x],[constraint_list_0[-1][0]]), \
                                        (pair['elements'][0]['radius']+pair['elements'][1]['radius'])**2-self.params.tol_obs,1e6+self.params.tol_obs])
                # constraint_list_0[-1][1] += self.params.collision_margin**2
                constraint_list_1_N_minus_1.append(constraint_list_0[-1])
                constraint_list_N.append(constraint_list_0[-1])

            if pair['type'] == 'sphere-plane':
                constr_expr = plane_sphere_dist(pair['elements'][1],pair['elements'][0]['fk'])
                constraint_list_0.append([constr_expr,pair['elements'][1]['bounds'][0] + pair['elements'][0]['radius']+self.params.collision_margin*2,pair['elements'][1]['bounds'][1] - pair['elements'][0]['radius']-self.params.collision_margin*2])
                self.collisions_constr_fun.append([cs.Function(f"collision_constraint_{i}_{pair['elements'][0]['name']}_{pair['elements'][1]['name']}",[self.x],[constraint_list_0[-1][0]]), \
                                        pair['elements'][1]['bounds'][0] + pair['elements'][0]['radius']-self.params.tol_obs,pair['elements'][1]['bounds'][1] - pair['elements'][0]['radius']+self.params.tol_obs])
                # constraint_list_0[-1][1] += self.params.collision_margin
                # constraint_list_0[-1][2] -= self.params.collision_margin
                constraint_list_1_N_minus_1.append(constraint_list_0[-1])
                constraint_list_N.append(constraint_list_0[-1])

       
        return constraint_list_0,constraint_list_1_N_minus_1,constraint_list_N
    
    def update_randomized_dynamics(self,controller_name=''):
        # Model with noise
        self.kin_dyn_noisy = KinDynComputations(self.params.robot_urdf[:-5] + f'_randomized{controller_name}.urdf', self.joint_names, self.params.robot_descr.get_root())        
        self.kin_dyn_noisy.set_frame_velocity_representation(adam.Representations.MIXED_REPRESENTATION)
        self.mass_noisy = self.kin_dyn_noisy.mass_matrix_fun()                           # Mass matrix
        self.bias_noisy = self.kin_dyn_noisy.bias_force_fun()                            # Nonlinear effects  
        self.gravity_noisy = self.kin_dyn_noisy.gravity_term_fun()                       # Gravity vector
        self.fk_noisy = self.kin_dyn_noisy.forward_kinematics_fun(self.params.frame_name)     # Forward kinematics

    def reset_seed(self,seed):
        self.rng = np.random.default_rng(seed=seed) 


class QuadrotorModel(AdamModel):
    def __init__(self, params):
        self.params = params
        self.amodel = AcadosModel()
        self.amodel.name = 'quadrotor'

        # Positions 
        x = MX.sym('x')
        y = MX.sym('y')
        z = MX.sym('z')

        # # Quarternion heading (body frame )
        q1 = MX.sym('q1')
        q2 = MX.sym('q2')
        q3 = MX.sym('q3')
        q4 = MX.sym('q4')

        # Transaltional velocities (inertial frame, m/s)
        vx = MX.sym('vx')
        vy = MX.sym('vy')
        vz = MX.sym('vz')
        v_c = vertcat(vx, vy, vz)

        # Angular velocities w.r.t phi(roll), theta(pitch), psi(yaw)
        # (body frame, m/s)
        wr = MX.sym('wr')
        wp = MX.sym('wp')
        wy = MX.sym('wy')
        omg = vertcat(wr, wp, wy)

        # Control variable angles (Motor RPM)
        ohm1 = MX.sym('ohm1')
        ohm2 = MX.sym('ohm2')
        ohm3 = MX.sym('ohm3')
        ohm4 = MX.sym('ohm4')

        self.x = vertcat(x, y, z, q1, q2, q3, q4, wr, wp, wy, vx, vy, vz)
        self.u = vertcat( ohm1, ohm2, ohm3, ohm4)

        mq = params.mass
        l = params.arm_length
        Ct = params.Ct
        Cd = params.Cd
        g0 = 9.81
        J = params.J

        D = (Cd / mq) *vertcat(vx*2, vy*2, vz**2)
        F = Ct * vertcat(0, 0, ohm1**2  + ohm2**2  + ohm3**2  + ohm4**2 )
        G = vertcat(0, 0, g0)
        J = params.J
        M = vertcat(Ct * l * (ohm1**2 + ohm2**2 - ohm3**2 - ohm4**2),
                      Ct * l * (ohm1**2 - ohm2**2 - ohm3**2 + ohm4**2),
                      Cd * (ohm1**2 - ohm2**2 + ohm3**2 - ohm4**2))

        Rq = vertcat(
            cs.horzcat( 2 * (q1**2 + q2**2) - 1,    -2 * (q1*q4 - q2*q3),       2 * (q1*q3 + q2*q4)),
            cs.horzcat( 2 * (q1*q4 + q2*q3),         2 * (q1**2 + q3**2) - 1,   2 * (q1*q2 - q3*q4)),
            cs.horzcat( 2 * (q1*q3 - q2*q4),         2 * (q1*q2 + q3*q4),       2 * (q1**2 + q4**2) - 1)
        )

        # Orientation ODEs ( qauternion)
        q1Dot = (-(q2 * wr) - (q3 * wp) - (q4 * wy))/2
        q2Dot = ( (q1 * wr) - (q4 * wp) + (q3 * wy))/2
        q3Dot = ( (q4 * wr) + (q1 * wp) - (q2 * wy))/2
        q4Dot = (-(q3 * wr) + (q2 * wp) + (q1 * wy))/2

        # Cartesian velocity ODEs ( including drag)
        vDot_c = -G + (1/ mq) * Rq @ F - D

        # Angular velocity ODEs (rate of change of projected Euler angles)
        omgDot = cs.inv(J) @ (M - cs.cross(omg, J @ omg))

        
        self.f_expl = vertcat(vx, vy, vz, 
            q1Dot, q2Dot, q3Dot, q4Dot,
            omgDot[0], omgDot[1], omgDot[2],
            vDot_c[0], vDot_c[1] , vDot_c[2]
        )

        self.f_fun = Function('f', [self.x, self.u], [self.f_expl])

        self.amodel.x = self.x
        self.amodel.u = self.u
        self.amodel.f_expl_expr = self.f_expl

        self.nx = self.amodel.x.size()[0]
        self.nu = self.amodel.u.size()[0]
        self.ny = self.nx + self.nu
        self.nq = 7
        self.nv = 6

        #TODO: Noise dynamics and inverse dyn

        # Limits, define those parameters
        self.x_min = - 1e6 * np.ones(self.nx)
        self.x_max = 1e6 * np.ones(self.nx)
        self.bounds_diff = np.abs(self.x_max-self.x_min)

        self.u_min = np.zeros(self.nu)
        self.u_max = np.array(params.thrust_limits)

        self.num_bound_u = 1

        # EE ref (position of the drone itself)
        self.t_glob = self.x[:3]
        self.ee_ref = self.params.ee_ref

        # Cartesian constraints
        self.obs_string = self.params.obs_string
        n_cap=0

        for sphere in self.params.spheres_robot:
            # The EE is the drone itself
            T_sphere = MX.eye(4)
            T_sphere[:3, 3] = self.x[:3]        # Drone position
            sphere['fk'] = T_sphere[:3,3] +T_sphere[:3, :3]@ sphere['spatial_offset']      
            sphere['fk_fun'] = Function(f'sphere_fk_{n_cap}', [self.x], [sphere['fk']])
            sphere['index']=n_cap
            n_cap +=1

        self.NL_external = self.generate_NLconstraints_list()

        # Integrator
        sim = AcadosSim()
        sim.model = self.amodel
        sim.solver_options.T = params.dt
        sim.solver_options.num_stages = 4       # ERK 4 
        self.acados_integrator = AcadosSimSolver(sim)

    def checkStateConstraints(self, x):
        return np.all(np.logical_and(x >= self.x_min - self.params.tol_x, 
                                     x <= self.x_max + self.params.tol_x))
    def checkInputConstraints(self, x, u):
        return np.all(np.logical_and(u >= self.u_min - self.params.tol_tau, 
                                     u <= self.u_max + self.params.tol_tau))
    
    def checkTorqueBounds(self, tau):
        raise NotImplementedError
    
    def checkRunningConstraints(self, x, u):
        return self.checkStateConstraints(x) and self.checkInputConstraints(x,u)

    def integrate(self, x, u):
        self.acados_integrator.set('x', x)
        self.acados_integrator.set('u', u)
        self.acados_integrator.solve()
        return self.acados_integrator.get('x'), u
    
    def integrate_controller_model(self, x, u):
        x_next, _ = self.integrate(x, u)
        return x_next, u
    
    def integrate_naively(self, x, u):
        return self.integrate(x, u)
    
    def generate_NLconstraints_list(self):
        """
        Generate list of nonlinear constraints with bounds, for nodes 0, 1 - N-1, and N, as well as the list of casadi function of the collision constraints
        """
        constraint_list_0 = []
        constraint_list_1_N_minus_1 = []
        constraint_list_N = []

        # generate also list of collision function for collision checks
        self.collisions_constr_fun = []

        # collisions
        for i,pair in enumerate(self.params.collisions_pairs):        
            if pair['type'] == 'sphere-sphere':
                #constr_expr = sphere_sphere_dist(pair['elements'][1],pair['elements'][0]['fk'])
                constr_expr = (self.t_glob - pair['elements'][1]['position']).T @ (self.t_glob - pair['elements'][1]['position'])
                constraint_list_0.append([constr_expr, (pair['elements'][0]['radius']+pair['elements'][1]['radius'] + self.params.collision_margin*2)**2,1e6])
                self.collisions_constr_fun.append([cs.Function(f"collision_constraint_{i}_{pair['elements'][0]['name']}_{pair['elements'][1]['name']}",[self.x],[constraint_list_0[-1][0]]), \
                                        (pair['elements'][0]['radius']+pair['elements'][1]['radius'])**2-self.params.tol_obs,1e6+self.params.tol_obs])
                # constraint_list_0[-1][1] += self.params.collision_margin**2
                constraint_list_1_N_minus_1.append(constraint_list_0[-1])
                constraint_list_N.append(constraint_list_0[-1])

            if pair['type'] == 'sphere-plane':
                constr_expr = plane_sphere_dist(pair['elements'][1],pair['elements'][0]['fk'])
                constraint_list_0.append([constr_expr,pair['elements'][1]['bounds'][0] + pair['elements'][0]['radius']+self.params.collision_margin*2,pair['elements'][1]['bounds'][1] - pair['elements'][0]['radius']-self.params.collision_margin*2])
                self.collisions_constr_fun.append([cs.Function(f"collision_constraint_{i}_{pair['elements'][0]['name']}_{pair['elements'][1]['name']}",[self.x],[constraint_list_0[-1][0]]), \
                                        pair['elements'][1]['bounds'][0] + pair['elements'][0]['radius']-self.params.tol_obs,pair['elements'][1]['bounds'][1] - pair['elements'][0]['radius']+self.params.tol_obs])
                # constraint_list_0[-1][1] += self.params.collision_margin
                # constraint_list_0[-1][2] -= self.params.collision_margin
                constraint_list_1_N_minus_1.append(constraint_list_0[-1])
                constraint_list_N.append(constraint_list_0[-1])

       
        return constraint_list_0,constraint_list_1_N_minus_1,constraint_list_N
    