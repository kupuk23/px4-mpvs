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
from px4_mpvs.utils.math_utils import quat_mul_cs


class SpacecraftVSModel:
    def __init__(self):

        self.name = "spacecraft_mpvs_model"

        # Camera intrinsic parameters
        self.K = np.array(
            [
                [443.53, 0.0, 320.0],  # fx, 0, cx
                [0.0, 443.53, 240.0],  # 0, fy, cy
                [0.0, 0.0, 1.0],  # 0, 0, 1
            ]
        )
        self.K = cs.DM(self.K)  # Convert to CasADi DM for compatibility

        # constants
        self.mass = 16.8
        self.inertia = np.diag((0.1454, 0.1366, 0.1594))
        self.max_thrust = 1.5
        self.max_rate = 0.5
        self.torque_arm_length = 0.12

        # BEARING-ERROR variables
        self.theta_max_deg = 10

    def get_interaction_matrix(self, s: cs.MX, Z: cs.MX) -> cs.MX:
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

    def adj_transform_inverse(self, R, p):
        Rt = R.T
        return cs.vertcat(
            cs.horzcat(Rt, -cs.mtimes(Rt, self.skew_symmetric_3x3(p))),
            cs.horzcat(cs.DM.zeros(3, 3), Rt),
        )

    def get_feature_dynamics(self, L, p, v, q, w):
        # Image feature dynamics equation

        # transform twist from map to camera frame.
        twist_map = cs.vertcat(v, w)  # 6x1
        # twist_map -> twist_base:
        R_mb = self.q_to_rot_mat(q)  # rotation matrix from map to base frame
        twist_base = cs.mtimes(self.adj_transform_inverse(R_mb, p), twist_map)  # 6x6

        # twist_base -> twist_cam:
        R_bc = cs.DM([[-1, 0, 0], [0, -1, 0], [0, 0, 1]])
        t_bc = cs.DM([-0.09, 0.0, 0.51])  # camera translation from base frame
        twist_cam = cs.mtimes(self.adj_transform_inverse(R_bc, t_bc), twist_base)  # 6x6

        # convert to openCV camera frame reference, x_im -> -y_cam, y_im -> -z_cam, z_im -> x_cam
        R_cam_to_optical = cs.DM([
            [0, -1,  0],  # x_optical = -y_cam
            [0,  0, -1],  # y_optical = -z_cam  
            [1,  0,  0]   # z_optical = x_cam
        ])

        # Transform the twist properly
        v_cam = twist_cam[:3]
        w_cam = twist_cam[3:]
        v_optical = cs.mtimes(R_cam_to_optical, v_cam)
        w_optical = cs.mtimes(R_cam_to_optical, w_cam)
        twist_optical = cs.vertcat(v_optical, w_optical)

        s_dot_vec = cs.mtimes(L, twist_optical)  # 8x1
        return s_dot_vec

    def build_interaction_mat_fun(self):
        s_sym = cs.MX.sym("s_fun", 8)  # image-Jacobian
        z_sym = cs.MX.sym("Z_fun", 4)  # linear twist
        L_sym = self.get_interaction_matrix(s_sym, z_sym)

        # Now hand it to CasADi
        self.L_f = cs.Function("L_fun", [s_sym, z_sym], [L_sym], ["s", "Z"], ["L"])

    def build_feature_dyn_fun(self):
        L_sym = cs.MX.sym("L", 8, 6)  # image-Jacobian
        v_sym = cs.MX.sym("v", 3)  # linear twist
        w_sym = cs.MX.sym("w", 3)  # angular twist
        q_sym = cs.MX.sym("q", 4)  # quaternion
        p_sym = cs.MX.sym("p", 3)  # position in inertial frame
        s_dot_sym = self.get_feature_dynamics(L_sym, p_sym, v_sym, q_sym, w_sym)

        # Now hand it to CasADi
        self.feat_dyn_f = cs.Function(
            "feat_dyn",
            [L_sym, p_sym, v_sym, q_sym, w_sym],
            [s_dot_sym],
            ["L", "p", "v", "q", "w"],
            ["s_dot"],
        )

    def build_debug_functions(self):
        """Build debug functions for twist transformations"""
        p_sym = cs.MX.sym("p", 3)
        v_sym = cs.MX.sym("v", 3)
        q_sym = cs.MX.sym("q", 4)
        w_sym = cs.MX.sym("w", 3)

        # Create twist transformations step by step
        twist_map = cs.vertcat(v_sym, w_sym)

        # Map to base transformation
        R_mb = self.q_to_rot_mat(q_sym)
        twist_base = cs.mtimes(self.adj_transform_inverse(R_mb, p_sym), twist_map)

        # Base to camera transformation
        R_bc = cs.DM([[-1, 0, 0], [0, -1, 0], [0, 0, 1]])
        t_bc = cs.DM([-0.09, 0.0, 0.51])
        twist_cam = cs.mtimes(self.adj_transform_inverse(R_bc, t_bc), twist_base)

        twist_optical = cs.vertcat(
            -twist_cam[1],
            -twist_cam[2],
            twist_cam[0],
            -twist_cam[4],
            -twist_cam[5],
            twist_cam[3],
        )

        # Create debug functions
        self.debug_twist_map = cs.Function(
            "debug_twist_map", [v_sym, w_sym], [twist_map], ["v", "w"], ["twist_map"]
        )

        self.debug_twist_base = cs.Function(
            "debug_twist_base",
            [p_sym, v_sym, q_sym, w_sym],
            [twist_map, twist_base, R_mb],
            ["p", "v", "q", "w"],
            ["twist_map", "twist_base", "R_mb"],
        )

        self.debug_twist_cam = cs.Function(
            "debug_twist_cam",
            [p_sym, v_sym, q_sym, w_sym],
            [twist_map, twist_base, twist_cam, twist_optical],
            ["p", "v", "q", "w"],
            ["twist_map", "twist_base", "twist_cam", "twist_optical"],
        )

        # # Debug adjoint transformations
        # self.debug_adj_mb = cs.Function(
        #     "debug_adj_mb",
        #     [q_sym, p_sym],
        #     [self.adj_transform_inverse(R_mb, p_sym)],
        #     ["q", "p"],
        #     ["adj_mb"]
        # )

        # self.debug_adj_bc = cs.Function(
        #     "debug_adj_bc",
        #     [],
        #     [self.adj_transform_inverse(R_bc, t_bc)],
        #     [],
        #     ["adj_bc"]
        # )

    def skew_symmetric(self, v):
        return cs.vertcat(
            cs.horzcat(0, -v[0], -v[1], -v[2]),
            cs.horzcat(v[0], 0, v[2], -v[1]),
            cs.horzcat(v[1], -v[2], 0, v[0]),
            cs.horzcat(v[2], v[1], -v[0], 0),
        )

    def skew_symmetric_3x3(self, v):
        return cs.vertcat(
            cs.horzcat(0, -v[2], v[1]),
            cs.horzcat(v[2], 0, -v[0]),
            cs.horzcat(-v[1], v[0], 0),
        )

    def q_to_rot_mat(self, q):
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

    def get_acados_model(self) -> AcadosModel:

        def v_dot_q(v, q):
            rot_mat = self.q_to_rot_mat(q)

            return cs.mtimes(rot_mat, v)

        def define_visual_constraint(p_obj, p, q):

            # rotate q 180 degrees around the z-axis
            # This is for the camera frame being reversed
            # q = [cos(θ/2), 0, 0, sin(θ/2)]

            quat_rot_z = cs.DM(
                [0, 0, 0, 1]
            )  # Quaternion representing 180 degrees rotation around z-axis
            q_rotated = quat_mul_cs(q, quat_rot_z)

            r_I = p_obj - p  # Vector from robot to object in inertial frame

            # Transform to body frame using the rotation matrix
            r_B = cs.mtimes(cs.transpose(self.q_to_rot_mat(q_rotated)), r_I)

            # Compute vector norm
            r_B_norm = cs.sqrt(
                r_B[0] ** 2 + r_B[1] ** 2 + r_B[2] ** 2
            )  # More explicit form
            cos_theta_max = cs.cos(np.deg2rad(self.theta_max_deg))
            g_x = cos_theta_max * r_B_norm - r_B[0]

            return g_x

        model = AcadosModel()

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
        w_s = cs.MX.sym("w_s", 1)  # Feature dynamics

        x = cs.vertcat(p, v, q, w, s)

        # compute bearing inequality

        g_x = define_visual_constraint(p_obj, p, q)

        # Define nonlinear constraint
        model.con_h_expr = g_x
        model.con_h_expr_e = g_x

        # Add model parameters
        model_params = cs.vertcat(p_obj, Z, w_p, w_s)  #

        # Define the image dynamics
        L = self.get_interaction_matrix(s, Z)

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
            1 / 2 * self.skew_symmetric(w) @ q,
            np.linalg.inv(self.inertia) @ (tau - cs.cross(w, self.inertia @ w)),
            self.get_feature_dynamics(L, p, v, q, w),
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
