import numpy as np
import casadi as ca
from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver, AcadosSimSolver
import scipy.linalg


# CrazyFlie 2.1 physical parameters
g0  = 9.81       # [m.s^2] gravitational accerelation
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
T_del = 0.005             # time between steps in seconds
N = 100                     # number of shooting nodes
Tf = N * T_del

Tsim = 45
Nsim = int(Tsim * N / Tf)

U_MAX = 30.0                                              # [krpm]
U_HOV = int(np.sqrt(.25 * 1e6* mq * g0 /Ct)) /1000      #[krpm]
U_REF = np.array([U_HOV, U_HOV, U_HOV, U_HOV])

# State
n_states = 13

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


dyn_f = ca.vertcat(vx, vy, vz, 
                    q1Dot, q2Dot, q3Dot, q4Dot,
                    omgDot[0], omgDot[1], omgDot[2],
                    vDot_c[0], vDot_c[1] , vDot_c[2], )
                #    ohm1Dot, ohm2Dot, ohm3Dot, ohm4Dot)

dyn_fun = ca.Function('f', [zeta_f, u], [dyn_f])


model = AcadosModel()
model.f_expl_expr = dyn_f
model.x = zeta_f
model.u = u
model.name = "ilporcoddidio"
ocp = AcadosOcp()
ocp.model = model

ocp.solver_options.N_horizon = N
ocp.solver_options.tf = Tf
nx = model.x.size()[0]
nu = model.u.size()[0]
ny = nx + nu

x0 = np.array([0.05, 0, 0,       # x,  y,  z
                      1, 0, 0, 0,       # qw, qx, qy, qz,
                      0, 0, 0,          # ohmr,  ohmp,  ohmy
                      0, 0, 0,  ]) 

# ocp.constraints.x0 = x0

ocp.cost.W = scipy.linalg.block_diag(Q, R)
ocp.cost.W_e = Qn

target = np.array([0.6, 0.3, 0.2])
ocp.cost.yref = np.zeros(ny)
ocp.cost.yref[:3] = target
ocp.cost.yref_e = np.zeros(nx)
ocp.cost.yref_e[:3] = target

ocp.cost.Vx = np.zeros((ny, nx))
ocp.cost.Vx[:nx, :nx] = np.eye(nx)

ocp.cost.Vx_e = np.eye(nx)

ocp.cost.Vu = np.zeros((ny, nu))
ocp.cost.Vu[-nu:, :] = np.eye(nu)

lbu = [0] * nu
ubu = [U_MAX] * nu


ocp.constraints.lbu = np.array(lbu)
ocp.constraints.ubu = np.array(ubu)
ocp.constraints.idxbu = np.array([0, 1, 2, 3])

ocp.constraints.lbx_0 = -np.ones_like(x0) * INF
ocp.constraints.ubx_0 = np.ones_like(x0) * INF
ocp.constraints.idxbx_0 = np.arange(nx)

ocp.constraints.lbx = np.array([-0.05, -0.05, -1.3])
ocp.constraints.ubx = np.array([INF, INF, INF])
ocp.constraints.idxbx = np.array([0, 1, 2])

ocp.solver_options.integrator_type = "ERK"
ocp.solver_options.nlp_solver_type = 'SQP'
# ocp.solver_options.hessian_approx =  "EXACT"   # do not work
ocp.solver_options.globalization = 'MERIT_BACKTRACKING'
ocp.solver_options.sim_method_num_stages = 4
ocp.solver_options.sim_method_num_steps = 1
ocp.solver_options.nlp_solver_max_iter = 300

solver = AcadosOcpSolver(ocp)
integrator = AcadosSimSolver(ocp)

x_guess = np.array([x0] * (N +1))
u0 = mq * g0 / 4 * np.ones(nu)
u_guess = np.array([u0] * N)
# verifica questo, probabilmente non è del tutto giusta questa guess
# verifica che basti dividere il peso per 4, ma il controllo credo 
# sia diverso nella dinamica

# Solve
solver.reset()
solver.set(0, 'lbx', x0)
solver.set(0, 'ubx', x0)

for i in range(N):
    solver.set(i, 'x', x_guess[i])
    solver.set(i, 'u', u_guess[i])
solver.set(N, 'x', x_guess[N])

solver.solve()
solver.print_statistics()

x_N = solver.get(N, 'x')

print(x_N)
print(np.linalg.norm(x_N[3:7]))

# IT WORKSSSS