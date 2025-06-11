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

from acados_template import AcadosModel
import casadi as cs
import numpy as np


class SpacecraftVSModel:
    def __init__(self):
        

        self.name = "spacecraft_mpavs_model"

        # Camera intrinsic parameters
        self.K = np.array(
            [
                [500.0, 0.0, 320.0],  # fx, 0, cx
                [0.0, 500.0, 240.0],  # 0, fy, cy
                [0.0, 0.0, 1.0],  # 0, 0, 1
            ]
        )

            

        # constants
        self.mass = 16.8
        self.inertia = np.diag((0.1454, 0.1366, 0.1594))
        self.max_thrust = 1.5
        self.max_rate = 0.5
        self.torque_arm_length = 0.12

        # BEARING-ERROR variables
        self.theta_max_deg = 10

    def _get_interaction_matrix(self, s: cs.MX, Z: cs.MX) -> cs.MX:
        L = cs.MX.zeros(s.shape[0], 6)

        N = int(s.shape[0] / 2)
        for i in range(N):
            x, y = s[i * 2], s[i * 2 + 1]
            depth = Z[i]
            # normalize the point
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

        return L  # cs.MX((8,6))

    def get_acados_model(self) -> AcadosModel:
        def skew_symmetric(v):
            return cs.vertcat(
                cs.horzcat(0, -v[0], -v[1], -v[2]),
                cs.horzcat(v[0], 0, v[2], -v[1]),
                cs.horzcat(v[1], -v[2], 0, v[0]),
                cs.horzcat(v[2], v[1], -v[0], 0),
            )

        def q_to_rot_mat(q):
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

        def v_dot_q(v, q):
            rot_mat = q_to_rot_mat(q)

            return cs.mtimes(rot_mat, v)

        def define_visual_constraint(p_obj, p, q):

            r_I = p_obj - p  # Vector from robot to object in inertial frame

            # Transform to body frame using the rotation matrix
            r_B = cs.mtimes(cs.transpose(q_to_rot_mat(q)), r_I)

            # Compute vector norm
            r_B_norm = cs.sqrt(
                r_B[0] ** 2 + r_B[1] ** 2 + r_B[2] ** 2
            )  # More explicit form
            cos_theta_max = cs.cos(np.deg2rad(self.theta_max_deg))
            g_x = cos_theta_max * r_B_norm - r_B[0]

            return g_x

        model = AcadosModel()

        def _get_feature_dynamics(L, v, w):
            # Image feature dynamics equation

            # transform twist from base to camera frame
            # w_cam = Rbc*w_base
            # v_cam = Rbc*(v_base + w_base x r_bc)
            # no rotation in the camera frame, Rbc = I
            r_bc = cs.DM([0.09, 0.0, 0.51])  # camera translation from base frame
            v_cam = v - cs.cross(w, r_bc)
            w_cam = w
            v_standard = cs.vertcat(
            -v_cam[1],  # X_std = -Y_your (right = -left)
            -v_cam[2],  # Y_std = -Z_your (down = -up)
             v_cam[0]   # Z_std = X_your (forward = forward)
        )
        
            w_standard = cs.vertcat(
                -w_cam[1],  # Roll around X_std = -pitch around Y_your
                -w_cam[2],  # Pitch around Y_std = -yaw around Z_your  
                w_cam[0]   # Yaw around Z_std = roll around X_your
            )
            twist = cs.vertcat(v_standard, w_standard)  # 6x1
            s_dot_vec = cs.mtimes(L, twist)  # 8x1
            # h_k = s + (s_dot * self.dt)  # 2x4 + 8x1
            return s_dot_vec
        
        # set up states & controls
        p = cs.MX.sym("p", 3)
        v = cs.MX.sym("v", 3)
        q = cs.MX.sym("q", 4)
        w = cs.MX.sym("w", 3)
        s = cs.MX.sym("s", 8)

        # Setup external parameters
        Z = cs.MX.sym("Z", 4)
        p_obj = cs.MX.sym("p_obj", 3)  # Object position in inertial frame
        w_p = cs.MX.sym("w_p", 1)  # Object angular velocity in inertial frame
        

        x = cs.vertcat(p, v, q, w, s)

        # compute bearing inequality
        
        g_x = define_visual_constraint(p_obj, p, q)

        # Define nonlinear constraint
        model.con_h_expr = g_x
        model.con_h_expr_e = g_x

        # Add model parameters
        model_params = cs.vertcat(p_obj, Z, w_p)  # 

        # Define the image dynamics
        L = self._get_interaction_matrix(s, Z)
        

        u = cs.MX.sym("u", 4)
        D_mat = cs.MX.zeros(2, 4)
        D_mat[0, 0] = 1
        D_mat[0, 1] = 1
        D_mat[1, 2] = -1
        D_mat[1, 3] = -1

        # L mat
        L_mat = cs.MX.zeros(1, 4)
        L_mat[0, 0] = -1
        L_mat[0, 1] = 1
        L_mat[0, 2] = -1
        L_mat[0, 3] = 1
        L_mat = L_mat * self.torque_arm_length

        F_2d = cs.mtimes(D_mat, u)
        tau_1d = cs.mtimes(L_mat, u)

        F = cs.vertcat(F_2d[0, 0], F_2d[1, 0], 0.0)
        tau = cs.vertcat(0.0, 0.0, tau_1d)

        # xdot
        p_dot = cs.MX.sym("p_dot", 3)
        v_dot = cs.MX.sym("v_dot", 3)
        q_dot = cs.MX.sym("q_dot", 4)
        w_dot = cs.MX.sym("w_dot", 3)
        s_dot = cs.MX.sym("s_dot", 8)

        xdot = cs.vertcat(p_dot, v_dot, q_dot, w_dot, s_dot)

        a_thrust = v_dot_q(F, q) / self.mass

        # dynamics
        f_expl = cs.vertcat(
            v,
            a_thrust,
            1 / 2 * cs.mtimes(skew_symmetric(w), q),
            np.linalg.inv(self.inertia) @ (tau - cs.cross(w, self.inertia @ w)),
            _get_feature_dynamics(L, v, w)
        )


        f_impl = xdot - f_expl

        model.f_impl_expr = f_impl
        model.f_expl_expr = f_expl
        model.x = x
        model.xdot = xdot
        model.u = u
        model.p = model_params 
        model.name = self.name

        return model
