############################################################################
#
#   Copyright (C) 2024 PX4 Development Team. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
# 1. Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in
#    the documentation and/or other materials provided with the
#    distribution.
# 3. Neither the name PX4 nor the names of its contributors may be
#    used to endorse or promote products derived from this software
#    without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
# COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS
# OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED
# AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
#
############################################################################

import casadi as cs
import numpy as np


class SpacecraftVSModelCasadi:
    def __init__(self, dt: float = 0.1):
        self.name = "spacecraft_mpvs_model_casadi"

        # Camera intrinsic parameters
        self.K = np.array(
            [
                [500.0, 0.0, 320.0],  # fx, 0, cx
                [0.0, 500.0, 240.0],  # 0, fy, cy
                [0.0, 0.0, 1.0],  # 0, 0, 1
            ]
        )
        self.K = cs.DM(self.K)  # Convert to CasADi DM for compatibility

        # Constants
        self.mass = 16.8
        self.inertia = cs.DM(np.diag((0.1454, 0.1366, 0.1594)))
        self.inertia_inv = cs.inv(self.inertia)
        self.max_thrust = 1.5
        self.max_rate = 0.5
        self.torque_arm_length = 0.12

        # Time step
        self.dt = dt
        
        # BEARING-ERROR variables
        self.theta_max_deg = 10
        self.cos_theta_max = np.cos(np.deg2rad(self.theta_max_deg))

        # State dimensions
        self.nx = 21  # p(3) + v(3) + q(4) + w(3) + s(8)
        self.nu = 4  # u(4) - thrust forces

        # Build CasADi functions
        self._build_dynamics_function()
        self._build_constraint_function()
        self._build_interaction_matrix_function()
        self._build_feature_dynamics_function()
        self._build_rk4_function(self.dt)

    def get_interaction_matrix(self, s: cs.MX, Z: cs.MX) -> cs.MX:
        """Compute the interaction matrix for visual servoing"""
        L = cs.MX.zeros(s.shape[0], 6)

        N = int(s.shape[0] / 2)
        for i in range(N):
            x, y = s[i * 2], s[i * 2 + 1]
            depth = Z[i]
            # Normalize the point
            x_n = (x - self.K[0, 2]) / self.K[0, 0]
            y_n = (y - self.K[1, 2]) / self.K[1, 1]

            row = i * 2

            L[row, 0] = -1.0 / depth
            L[row, 1] = 0.0
            L[row, 2] = x_n / depth
            L[row, 3] = x_n * y_n
            L[row, 4] = -(1.0 + x_n * x_n)
            L[row, 5] = y_n

            # For vy
            L[row + 1, 0] = 0.0
            L[row + 1, 1] = -1.0 / depth
            L[row + 1, 2] = y_n / depth
            L[row + 1, 3] = 1.0 + y_n * y_n
            L[row + 1, 4] = -x_n * y_n
            L[row + 1, 5] = -x_n

        return L

    def get_feature_dynamics(self, L, v, w):
        """Compute image feature dynamics"""
        # Transform twist from base to camera frame
        Rbc = cs.DM([[-1, 0, 0], [0, -1, 0], [0, 0, 1]])
        r_bc = cs.DM([-0.09, 0.0, 0.51])

        v_cam = cs.mtimes(Rbc, (v + cs.cross(w, r_bc)))
        w_cam = cs.mtimes(Rbc, w)

        v_image = cs.vertcat(
            v_cam[1],  # X_image = -Y_cam
            -v_cam[2],  # Y_image = -Z_cam
            -v_cam[0],  # Z_image = X_cam
        )

        w_image = cs.vertcat(
            w_cam[1],  # Roll around X_image
            -w_cam[2],  # Pitch around Y_image
            -w_cam[0],  # Yaw around Z_image
        )

        twist = cs.vertcat(v_image, w_image)
        s_dot = cs.mtimes(L, twist)
        return s_dot

    def skew_symmetric(self, v):
        """Create skew-symmetric matrix for quaternion dynamics"""
        return cs.vertcat(
            cs.horzcat(0, -v[0], -v[1], -v[2]),
            cs.horzcat(v[0], 0, v[2], -v[1]),
            cs.horzcat(v[1], -v[2], 0, v[0]),
            cs.horzcat(v[2], v[1], -v[0], 0),
        )

    def q_to_rot_mat(self, q):
        """Convert quaternion to rotation matrix"""
        qw, qx, qy, qz = q[0], q[1], q[2], q[3]

        rot_mat = cs.vertcat(
            cs.horzcat(
                1 - 2 * (qy**2 + qz**2),
                2 * (qx * qy - qw * qz),
                2 * (qx * qz + qw * qy),
            ),
            cs.horzcat(
                2 * (qx * qy + qw * qz),
                1 - 2 * (qx**2 + qz**2),
                2 * (qy * qz - qw * qx),
            ),
            cs.horzcat(
                2 * (qx * qz - qw * qy),
                2 * (qy * qz + qw * qx),
                1 - 2 * (qx**2 + qy**2),
            ),
        )
        return rot_mat

    def quat_mul(self, q1, q2):
        """Quaternion multiplication"""
        w1, x1, y1, z1 = q1[0], q1[1], q1[2], q1[3]
        w2, x2, y2, z2 = q2[0], q2[1], q2[2], q2[3]

        w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
        x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
        y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
        z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

        return cs.vertcat(w, x, y, z)

    def get_dynamics(self, x, u, params):
        """Compute system dynamics dx/dt = f(x, u, params)"""
        # Extract states
        p = x[0:3]  # position
        v = x[3:6]  # velocity
        q = x[6:10]  # quaternion
        w = x[10:13]  # angular velocity
        s = x[13:21]  # image features

        # Extract parameters
        p_obj = params[0:3]  # object position
        Z = params[3:7]  # depths

        # Control allocation matrices
        D_mat = cs.MX.zeros(2, 4)
        D_mat[0, 0] = 1
        D_mat[0, 1] = 1
        D_mat[1, 2] = -1
        D_mat[1, 3] = -1

        L_mat = cs.MX.zeros(1, 4)
        L_mat[0, 0] = -1
        L_mat[0, 1] = 1
        L_mat[0, 2] = -1
        L_mat[0, 3] = 1
        L_mat = L_mat * self.torque_arm_length

        # Compute forces and torques
        F_2d = cs.mtimes(D_mat, u)
        tau_1d = cs.mtimes(L_mat, u)
        F = cs.vertcat(F_2d[0], F_2d[1], 0.0)
        tau = cs.vertcat(0.0, 0.0, tau_1d[0])

        # Rotate force to inertial frame
        R = self.q_to_rot_mat(q)
        F_inertial = cs.mtimes(R, F)

        # Compute accelerations
        a = F_inertial / self.mass
        alpha = cs.mtimes(
            self.inertia_inv, (tau - cs.cross(w, cs.mtimes(self.inertia, w)))
        )

        # Quaternion derivative
        q_dot = 0.5 * cs.mtimes(self.skew_symmetric(w), q)

        # Feature dynamics
        L = self.get_interaction_matrix(s, Z)
        s_dot = self.get_feature_dynamics(L, v, w)

        # Assemble state derivative
        xdot = cs.vertcat(v, a, q_dot, alpha, s_dot)

        return xdot

    def rk4_integrator(self, x, u, params, dt):
        """RK4 integrator for spacecraft dynamics"""
        # Define the continuous dynamics function
        f = lambda x: self.get_dynamics(x, u, params)

        # RK4 steps
        k1 = f(x)
        k2 = f(x + dt / 2 * k1)
        k3 = f(x + dt / 2 * k2)
        k4 = f(x + dt * k3)

        # Compute next state
        x_next = x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

        return x_next

    def _build_rk4_function(self, dt):
        """Build CasADi function for RK4 integrator"""
        x_sym = cs.MX.sym("x", self.nx)
        u_sym = cs.MX.sym("u", self.nu)
        p_sym = cs.MX.sym("p", 7)

        x_next = self.rk4_integrator(x_sym, u_sym, p_sym, dt)

        self.f_rk4 = cs.Function(
            "rk4", [x_sym, u_sym, p_sym], [x_next], ["x", "u", "params"], ["x_next"]
        )

    def get_visual_constraint(self, x, params):
        """Compute bearing constraint g(x) <= 0"""
        # Extract states
        p = x[0:3]
        q = x[6:10]

        # Extract parameters
        p_obj = params[0:3]

        # Rotate quaternion 180 degrees around z-axis for camera frame
        quat_rot_z = cs.DM([0, 0, 0, 1])  # 180 deg rotation
        q_rotated = self.quat_mul(q, quat_rot_z)

        # Vector from robot to object in inertial frame
        r_I = p_obj - p

        # Transform to body frame
        R = self.q_to_rot_mat(q_rotated)
        r_B = cs.mtimes(cs.transpose(R), r_I)

        # Compute constraint
        r_B_norm = cs.sqrt(r_B[0] ** 2 + r_B[1] ** 2 + r_B[2] ** 2)
        g = self.cos_theta_max * r_B_norm - r_B[0]

        return g

    def _build_dynamics_function(self):
        """Build CasADi function for dynamics"""
        x_sym = cs.MX.sym("x", self.nx)
        u_sym = cs.MX.sym("u", self.nu)
        p_sym = cs.MX.sym("p", 7)  # p_obj(3) + Z(4)

        xdot = self.get_dynamics(x_sym, u_sym, p_sym)

        self.f_dynamics = cs.Function(
            "dynamics", [x_sym, u_sym, p_sym], [xdot], ["x", "u", "params"], ["xdot"]
        )

    def _build_constraint_function(self):
        """Build CasADi function for visual constraint"""
        x_sym = cs.MX.sym("x", self.nx)
        p_sym = cs.MX.sym("p", 7)

        g = self.get_visual_constraint(x_sym, p_sym)

        self.f_constraint = cs.Function(
            "visual_constraint", [x_sym, p_sym], [g], ["x", "params"], ["g"]
        )

    def _build_interaction_matrix_function(self):
        """Build CasADi function for interaction matrix"""
        s_sym = cs.MX.sym("s", 8)
        Z_sym = cs.MX.sym("Z", 4)

        L = self.get_interaction_matrix(s_sym, Z_sym)

        self.f_interaction = cs.Function(
            "interaction_matrix", [s_sym, Z_sym], [L], ["s", "Z"], ["L"]
        )

    def _build_feature_dynamics_function(self):
        """Build CasADi function for feature dynamics"""
        L_sym = cs.MX.sym("L", 8, 6)
        v_sym = cs.MX.sym("v", 3)
        w_sym = cs.MX.sym("w", 3)

        s_dot = self.get_feature_dynamics(L_sym, v_sym, w_sym)

        self.f_feature_dynamics = cs.Function(
            "feature_dynamics",
            [L_sym, v_sym, w_sym],
            [s_dot],
            ["L", "v", "w"],
            ["s_dot"],
        )
