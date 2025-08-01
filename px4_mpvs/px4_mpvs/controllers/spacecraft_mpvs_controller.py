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

from acados_template import AcadosOcp, AcadosOcpSolver, AcadosSimSolver
import numpy as np
import casadi as cs
import time
from px4_mpvs.utils.math_utils import quat_error
from px4_mpvs.models.spacecraft_vs_model import SpacecraftVSModel
from time import perf_counter


class SpacecraftVSMPC:
    def __init__(self, model, build=False, p_obj=None, x0=None, Z=None):

        self.build = build  # Set to False after the first run to avoid rebuilding
        self.vel_limit = 0.8  # np.inf .1
        self.model = model
        self.Tf = 5.0
        self.N = 24  # TODO: check how fast the update rate
        self.ibvs_mode = False  # True for ibvs, False for pbvs

        self.Qp_p = 1e2  # Position weights (x, y, z), # 5e1 pbvs, 0 for ibvs
        self.Qp_q = 3e3  # Quaternion scalar part, 8e3
        self.w_features = 4e-3  # Image feature weights, 0 pbvs, 5e-3 for ibvs

        self.x0 = (
            x0
            if x0 is not None
            else np.array(
                [
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    1.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    100,
                    100,
                    400,
                    100,
                    100,
                    300,
                    400,
                    300,
                ]
            )
        )

        # PBVS parameters
        self.ineq_bounds = 1e4
        self.w_slack = 0
        self.p_obj0 = p_obj if p_obj is not None else np.array([-10.0, 0.0, 0.0])

        # pixel bounds for image features
        self.s_min = 10
        self.s_max = 640 - 10  # 640x480 image

        self.Z0 = Z if Z is not None else np.array([1.0] * 4)

        self.ocp_solver, self.integrator = self.setup(
            self.x0, self.N, self.Tf, self.p_obj0, Z0=self.Z0
        )
        self.model.build_feature_dyn_fun()
        self.model.build_interaction_mat_fun()
        self.model.build_debug_functions()

    def set_constraints(self, ocp, x0):
        Fmax = self.model.max_thrust
        # constraint bounds -inf <= g(x) <= 0
        ocp.constraints.lh = np.array([-self.ineq_bounds])
        ocp.constraints.uh = np.array([0.0])
        ocp.constraints.lh_e = np.array([-self.ineq_bounds])
        ocp.constraints.uh_e = np.array([0.0])

        # define soft constraint using slack variable
        ocp.constraints.idxsh = np.array([0])
        ocp.constraints.idxsh_e = np.array([0])

        ocp.cost.Zl = np.array([0.0])
        ocp.cost.Zu = np.array([self.w_slack])
        ocp.cost.Zl_e = np.array([0.0])
        ocp.cost.Zu_e = np.array([self.w_slack])

        ocp.cost.zl = np.array([0.0])
        ocp.cost.zu = np.array([self.w_slack])
        ocp.cost.zl_e = np.array([0.0])
        ocp.cost.zu_e = np.array([self.w_slack])

        # set bounds for image features (x coordinates)
        # ocp.constraints.idxbx = np.array([13, 15, 17, 19])
        # ocp.constraints.lbx = np.array([self.s_min] * 4)
        # ocp.constraints.ubx = np.array([self.s_max] * 4)

        # set constraints
        ocp.constraints.lbu = np.array([-Fmax, -Fmax, -Fmax, -Fmax])
        ocp.constraints.ubu = np.array([+Fmax, +Fmax, +Fmax, +Fmax])
        ocp.constraints.idxbu = np.array([0, 1, 2, 3])
        ocp.constraints.x0 = x0

        # set constraints on X
        ocp.constraints.lbx = np.array(
            [
                -5,
                -5,
                -5,
                -self.vel_limit,
                -self.vel_limit,
                -self.vel_limit,
                -self.vel_limit,
                -self.vel_limit,
                -self.vel_limit,
            ]
        )
        ocp.constraints.ubx = np.array(
            [
                +5,
                +5,
                +5,
                +self.vel_limit,
                +self.vel_limit,
                +self.vel_limit,
                +self.vel_limit,
                +self.vel_limit,
                +self.vel_limit,
            ]
        )
        ocp.constraints.idxbx = np.array([0, 1, 2, 3, 4, 5, 10, 11, 12])

        # set constraints on X at the end of the horizon
        ocp.constraints.lbx_e = np.array(
            [
                -5,
                -5,
                -5,
                -self.vel_limit,
                -self.vel_limit,
                -self.vel_limit,
                -self.vel_limit,
                -self.vel_limit,
                -self.vel_limit,
            ]
        )
        ocp.constraints.ubx_e = np.array(
            [
                +5,
                +5,
                +5,
                self.vel_limit,
                self.vel_limit,
                self.vel_limit,
                self.vel_limit,
                self.vel_limit,
                self.vel_limit,
            ]
        )
        ocp.constraints.idxbx_e = np.array([0, 1, 2, 3, 4, 5, 10, 11, 12])

        use_soft_constraints = False
        if use_soft_constraints:
            # set weights slack variables for X constraints
            ocp.constraints.idxsbx = np.arange(len(ocp.constraints.idxbx))
            ocp.cost.Zl = np.array([1e6] * len(ocp.constraints.idxsbx))
            ocp.cost.Zu = np.array([1e6] * len(ocp.constraints.idxsbx))
            ocp.cost.zl = np.array([0.0] * len(ocp.constraints.idxsbx))
            ocp.cost.zu = np.array([0.0] * len(ocp.constraints.idxsbx))

            # set weights slack variables for X_e constraints
            ocp.constraints.idxsbx_e = np.arange(len(ocp.constraints.idxbx_e))
            ocp.cost.Zl_e = np.array([1e6] * len(ocp.constraints.idxsbx_e))
            ocp.cost.Zu_e = np.array([1e6] * len(ocp.constraints.idxsbx_e))
            ocp.cost.zl_e = np.array([0.0] * len(ocp.constraints.idxsbx_e))
            ocp.cost.zu_e = np.array([0.0] * len(ocp.constraints.idxsbx_e))

        return ocp

    def setup(self, x0, N_horizon, Tf, p_obj0, Z0):
        def to_DM(A):
            return cs.DM(A) if isinstance(A, np.ndarray) else A

        # create ocp object to formulate the OCP
        ocp = AcadosOcp()
        # set model
        model = self.model.get_acados_model()

        ocp.model = model

        nx = model.x.size()[0]
        nu = model.u.size()[0]
        print(f"nx: {nx}, nu: {nu}")

        # set dimensions
        ocp.dims.N = N_horizon
        ocp.solver_options.N_horizon = N_horizon

        # Initialize parameters
        p_obj = ocp.model.p[0:3]
        Z = ocp.model.p[3:7]
        w_p = ocp.model.p[7]
        w_s = ocp.model.p[8]  # Feature dynamics

        Qp_p = self.Qp_p  # Position weights (x, y, z), # 5e1 pbvs, 0 for ibvs
        Qp_q = self.Qp_q  # Quaternion scalar part, 8e3
        w_features = self.w_features  # Image feature weights, 0 pbvs, 5e-3 for ibvs

        # set weights for the cost function
        Q = [
            *[Qp_p] * 3,  # Position weights (x, y, z), # 5e1 pbvs, 0 for ibvs
            *[1e2] * 3,  # Velocity weights (vx, vy, vz) # 5e1 pbvs, 5e3 for ibvs
            # Qp_q,
            Qp_q,
            *[2e2] * 3,  # angular vel (ωx, ωy, ωz) # 5e1 pbvs, 8e2 for ibvs
        ]

        # Qs = [
        #         *[0] * 3,  # Position weights (x, y, z), # 5e1 pbvs, 0 for ibvs
        #         *[50e2] * 3,  # Velocity weights (vx, vy, vz) # 70e2
        #         0,
        #         *[4e3] * 3,  # angular vel (ωx, ωy, ωz) #5e3
        #     ]

        S = [
            *[w_features] * 8,  # Image feature weights, 0 pbvs, 5e-3 for ibvs
        ]

        Q_e = [element * 30 for element in Q]
        S_e = [element * 80 for element in S]

        R_mat = [1e1] * 4

        ocp.cost.W_0 = np.diag(Q + S + R_mat)
        ocp.cost.W = np.diag(Q + S + R_mat)
        ocp.cost.W_e = np.diag(Q_e + S_e)

        # References:
        x_ref = cs.MX.sym("x_ref", (nx, 1))  # 13 robot states + 8 features states
        u_ref = cs.MX.sym("u_ref", (nu, 1))

        # Calculate errors
        # x : p,v,q,w               , R9 x SO(3)
        # u : Fx,Fy,Fz,Mx,My,Mz     , R6

        x = ocp.model.x
        u = ocp.model.u

        # Error scaling for x_error
        # p : wp
        # v : 50-(49wp)
        # q : wp
        # w : 10-(9wp)
        # s : 1-wp
        v_scale = cs.sqrt(50 - (49 * w_p))  # Scale for velocity error
        w_scale = cs.sqrt(10 - (9 * w_p))  # Scale for angular velocity error
        s_scale = cs.sqrt(1.0 - w_p)  # Scale for feature error

        x_error = cs.sqrt(w_p) * (x[0:3] - x_ref[0:3])
        x_error = cs.vertcat(x_error, v_scale * (x[3:6] - x_ref[3:6]))
        x_error = cs.vertcat(
            x_error, cs.sqrt(w_p) * (1 - (x[6:10].T @ x_ref[6:10]) ** 2)
        )
        x_error = cs.vertcat(x_error, w_scale * (x[10:13] - x_ref[10:13]))
        x_error = cs.vertcat(x_error, s_scale * (x[13:] - x_ref[13:]))
        u_error = u - u_ref

        ocp.model.p = cs.vertcat(x_ref, u_ref, p_obj, Z, w_p, w_s)

        # define cost with parametric reference
        ocp.cost.cost_type = "NONLINEAR_LS"
        ocp.cost.cost_type_e = "NONLINEAR_LS"
        ocp.cost.cost_type_0 = "NONLINEAR_LS"

        ocp.model.cost_y_expr_0 = cs.vertcat(x_error, u_error)
        ocp.model.cost_y_expr = cs.vertcat(x_error, u_error)
        ocp.model.cost_y_expr_e = x_error

        ocp.cost.yref_0 = np.zeros(ocp.model.cost_y_expr_0.shape[0])
        ocp.cost.yref = np.zeros(ocp.model.cost_y_expr.shape[0])
        ocp.cost.yref_e = np.zeros(ocp.model.cost_y_expr_e.shape[0])

        ocp = self.set_constraints(ocp, x0)

        # set initial state
        ocp.constraints.x0 = x0

        p_0 = np.concatenate(
            (x0, np.zeros(nu), p_obj0, Z0, np.ones(1), np.zeros(1))
        )  # Z = feature depth
        ocp.parameter_values = p_0

        # set options
        ocp.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM"
        # PARTIAL_CONDENSING_HPIPM, FULL_CONDENSING_QPOASES, FULL_CONDENSING_HPIPM,
        # PARTIAL_CONDENSING_QPDUNES, PARTIAL_CONDENSING_OSQP, FULL_CONDENSING_DAQP
        ocp.solver_options.hessian_approx = "GAUSS_NEWTON"  # 'GAUSS_NEWTON', 'EXACT'
        ocp.solver_options.integrator_type = "ERK"

        ocp.solver_options.print_level = 0

        use_RTI = True
        if use_RTI:
            ocp.solver_options.nlp_solver_type = "SQP_RTI"
            ocp.solver_options.sim_method_num_stages = 4

            ocp.solver_options.sim_method_num_steps = 3
        else:
            ocp.solver_options.nlp_solver_type = "SQP"  # SQP_RTI, SQP

        # ocp.solver_options.qp_solver_cond_N = N_horizon
        # ocp.solver_options.print_level = 6

        # set prediction horizon
        ocp.solver_options.tf = Tf

        json_file = "acados_hybrid_vs_ocp.json"
        ocp_solver = AcadosOcpSolver(
            ocp,
            json_file=json_file,
            build=self.build,
            generate=self.build,
        )

        # create an integrator with the same settings as used in the OCP solver.
        acados_integrator = AcadosSimSolver(
            ocp,
            json_file=json_file,
            build=self.build,
            generate=self.build,
        )

        return ocp_solver, acados_integrator

    def define_lyapunov_weight(self, x: cs.MX, x_ref: cs.MX, Qp_p, Qp_q, w_feat, s_dot):
        x = x.reshape(-1, 1)  # Ensure x is a column vector
        x_ref = x_ref.reshape(-1, 1)  # Ensure x_ref is
        v = x[3:6]  # Velocity

        e_p = x[0:3] - x_ref[0:3]  # Position error

        w_diag = cs.vertcat(cs.DM([Qp_p, Qp_p, Qp_p]))

        Qp_V = cs.diag(w_diag)  # PBVS Qp matrix

        S = np.diag(
            [
                *[w_feat] * 8,  # Image feature weights, 0 pbvs, 5e-3 for ibvs
            ]
        )

        S = S * 25

        e_s = x[13:] - x_ref[13:]

        Vp_dot = cs.mtimes([e_p.T, Qp_V, v])
        Vs_dot = cs.mtimes([e_s.T, S, s_dot])
        # softmax_p = 0
        # softmax_s = 0

        k = 3  # how sharp the softmax is, 3.5 for softmax mode

        softmax_p = cs.exp(-k * Vp_dot)
        softmax_p = cs.if_else(Vp_dot > 0.02, 0, softmax_p)
        softmax_s = cs.exp(-k * Vs_dot)
        eps = 1e-5  # keeps denominator strictly positive

        # # softmax weights
        w_s = softmax_s / (softmax_p + softmax_s + eps)
        w_p = 1.0 - w_s

        # Ratio method
        Vs_dot = cs.if_else(Vs_dot >= 0, 0, Vs_dot)
        w_p = cs.if_else(
            Vp_dot > 0, 0, Vp_dot / (Vp_dot + Vs_dot + eps)
        )  # ensure w_p is non-negative
        # w_p = cs.fmax(w_p, 0)  # ensure w_p is non-negative
        w_s = 1.0 - w_p  # w_s is always non-negative

        V_dot = Vp_dot + Vs_dot

        # convert to numpy arrays
        w_s = w_s.full().flatten()
        w_p = w_p.full().flatten()
        Vp_dot = Vp_dot.full().flatten()
        Vs_dot = Vs_dot.full().flatten()

        # self.lyapunov_eval = cs.Function(
        #     "lyapunov_eval",
        #     [x, x_ref, s_dot],
        #     [Vp_dot, Vs_dot, V_dot, w_p, w_s, softmax_p, softmax_s],
        #     ["state", "reference", "s_dot"],
        #     ["Vp_dot", "Vs_dot", "V_dot", "w_p", "w_s", "softmax_p", "softmax_s"],
        # )
        return w_p, w_s, Vp_dot, Vs_dot, V_dot, softmax_p, softmax_s

    def update_constraints(self, servoing_enabled):
        # Update the constraints based on the servoing_enabled flag
        self.w_slack = (
            2e3 if servoing_enabled else 0
        )  # 2e3, TODO: set to 0 for no slack

    def debug_twist_transformations(self, x0, verbose=True):
        """Debug twist transformations step by step"""
        p = x0[0:3]
        v = x0[3:6]
        q = x0[6:10]
        w = x0[10:13]
        
        # if verbose:
        #     print("="*50)
        #     print("TWIST TRANSFORMATION DEBUG")
        #     print("="*50)
        #     print(f"Position (p): {p}")
        #     print(f"Velocity (v): {v}")
        #     print(f"Quaternion (q): {q}")
        #     print(f"Angular velocity (w): {w}")
        #     print("-"*30)
        
        # Debug twist map
        twist_map = self.model.debug_twist_map(v, w)
        twist_map_np = twist_map.full().flatten()
        
        # Debug twist transformations
        twist_map_dbg, twist_base_dbg, R_mb = self.model.debug_twist_base(p, v, q, w)
        twist_map_np = twist_map_dbg.full().flatten()
        twist_base_np = twist_base_dbg.full().flatten()
        R_mb_np = R_mb.full()
        
        # Debug full transformation
        twist_map_full, twist_base_full, twist_cam_full, twist_optical_full = self.model.debug_twist_cam(p, v, q, w)
        twist_cam_np = twist_cam_full.full().flatten()
        twist_optical_np = twist_optical_full.full().flatten()

        # Debug adjoint matrices
        # adj_mb = self.model.debug_adj_mb(q, p)
        # adj_bc = self.model.debug_adj_bc()
        # adj_mb_np = adj_mb.full()
        # adj_bc_np = adj_bc.full()
        
        if verbose:
            print("="*50)
            print(f"Twist Map (v,w): [{', '.join([f'{x:.2f}' for x in twist_map_np])}]")
            print(f"Twist Base: [{', '.join([f'{x:.2f}' for x in twist_base_np])}]")
            print("-"*30)
            # print(f"Twist Camera: [{', '.join([f'{x:.2f}' for x in twist_cam_np])}]")
            print(f"Twist Optical: [{', '.join([f'{x:.2f}' for x in twist_optical_np])}]")
            print("="*50)

    def solve(self, x0, verbose=False, ref=None, p_obj=None, Z=None, hybrid_mode=False):

        # Set reference, create zero reference
        if ref is None:
            zero_ref = np.zeros(
                self.model.get_acados_model().x.size()[0]
                + self.model.get_acados_model().u.size()[0]
            )
            zero_ref[6] = 1.0

        # preparation phase
        ocp_solver = self.ocp_solver

        x_ref = ref[:-4, 0] if ref is not None else zero_ref[:-4, 0]

        # ref[6:10, :] = self.debug_quaternion_reference(x0, ref[:, 0])
        # q_err = quat_error(x0[6:10], ref[6:10,0])  # Quaternion error
        # # print the quaternion error for debugging
        # print(f"Quaternion error: {q_err}")

        p_obj = p_obj if p_obj is not None else self.p_obj0
        Z = Z if Z is not None else self.Z0

        L_val = self.model.L_f(x0[13:], Z)
        s_dot = self.model.feat_dyn_f(L_val, x0[0:3], x0[3:6], x0[6:10], x0[10:13])
        s_dot = s_dot.full().flatten()  # Convert to numpy array

        w_p, w_s, Vp_dot, Vs_dot, V_dot, softmax_p, softmax_s = (
            self.define_lyapunov_weight(
                ocp_solver.get(0, "x"),
                x_ref,
                self.Qp_p,
                self.Qp_q,
                self.w_features,
                s_dot,
            )
        )

        if hybrid_mode and not self.ibvs_mode:
            # TEST DISCRETE
            w_p = np.zeros(1)
            w_s = np.ones(1)

            if w_p < 0.05:
                self.ibvs_mode = True
        elif hybrid_mode and self.ibvs_mode:
            w_p = np.zeros(1)
            w_s = np.ones(1)
        else:
            self.ibvs_mode = False
            s_dot = np.zeros(8)
            w_p = np.ones(1)  # Set to 1 for no hybrid mode
            w_s = np.zeros(1)

        for i in range(self.N + 1):
            if i != self.N and i != 0:
                self.ocp_solver.cost_set(i, "Zu", np.array([self.w_slack]))
                self.ocp_solver.cost_set(i, "zu", np.array([self.w_slack]))

            if ref is not None:
                # Assumed ref structure: (nx+nu) x N+1
                # NOTE: last u_ref is not used
                p_i = np.concatenate([ref[:, i], p_obj, Z, w_p, w_s])
            else:
                # set all references to 0
                p_i = np.concatenate([zero_ref, p_obj, Z, w_p, w_s])
            ocp_solver.set(i, "p", p_i)

        # set initial state
        ocp_solver.set(0, "lbx", x0.flatten())
        ocp_solver.set(0, "ubx", x0.flatten())

        status = ocp_solver.solve()

        print(f"===== Lyapunov Values =====")
        # print(f"Vp: {float(Vp):.4f}, Vs: {float(Vs):.4f}")
        print(f"Vp_dot: {float(Vp_dot):.2f}, Vs_dot: {float(Vs_dot):.2f}")
        print(f"softmax_p: {float(softmax_p):.2f}, softmax_s: {float(softmax_s):.2f}")
        print(f"wp: {float(w_p):.2f}, ws: {float(w_s):.2f}")

        if verbose:
            # self.debug_twist_transformations(x0)
            if hybrid_mode and not self.ibvs_mode:
                # print(f"===== Lyapunov Values =====")
                # # print(f"Vp: {float(Vp):.4f}, Vs: {float(Vs):.4f}")
                # print(f"Vp_dot: {float(Vp_dot):.2f}, Vs_dot: {float(Vs_dot):.2f}")
                # print(
                #     f"softmax_p: {float(softmax_p):.2f}, softmax_s: {float(softmax_s):.2f}"
                # )
                # print(f"wp: {float(w_p):.2f}, ws: {float(w_s):.2f}")
                pass

            # ocp_solver.dump_last_qp_to_json(filename="last_qp.json", overwrite=True)

            # self.ocp_solver.print_statistics()  # encapsulates: stat = ocp_solver.get_stats("statistics")

        if status != 0:
            raise Exception(f"acados returned status {status}.")

        N = self.N
        nx = self.model.get_acados_model().x.size()[0]
        nu = self.model.get_acados_model().u.size()[0]

        simX = np.ndarray((N + 1, nx))
        simU = np.ndarray((N, nu))

        # get solution
        for i in range(N):
            simX[i, :] = self.ocp_solver.get(i, "x")
            simU[i, :] = self.ocp_solver.get(i, "u")
        simX[N, :] = self.ocp_solver.get(N, "x")

        return simU, simX, w_p, w_s, Vp_dot, Vs_dot
