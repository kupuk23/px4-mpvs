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

import numpy as np
import casadi as ca
import casadi.tools as ctools
import time
from time import perf_counter
from px4_mpvs.utils.math_utils import quat_error


class SpacecraftVSMPCCasadi:
    """
    Spacecraft Visual Servoing MPC Controller using CasADi/IPOPT
    Following the structure of spiraling_mpc.py

    Implements hybrid PBVS/IBVS control:
    - PBVS: Position-Based Visual Servoing (tracks 3D position)
    - IBVS: Image-Based Visual Servoing (tracks 2D image features)
    - Hybrid: Lyapunov-based switching between PBVS and IBVS
    """

    def __init__(self, model, p_obj=None, x0=None, Z=None):
        """
        Initialize the MPC controller using CasADi/IPOPT

        Args:
            model: SpacecraftVSModelCasadi instance
            p_obj: Object position
            x0: Initial state
            Z: Feature depths
        """
        self.model = model

        # Controller parameters
        self.vel_limit = 0.8
        self.Tf = 5.0
        self.N = 24  # Horizon
        self.ibvs_mode = False

        # State and control dimensions
        self.Nx = model.nx  # Full state dimension
        self.Nu = model.nu  # Control dimension
        self.Nopt = 13  # Optimized states (exclude features for cost)

        # Cost weights
        self.Qp_p = 5e1  # Position weights
        self.Qp_q = 8e2  # Quaternion weights
        self.w_features = 5e-3  # Image feature weights

        # Default initial state
        self.x0 = (
            x0
            if x0 is not None
            else np.array(
                [
                    0.0,
                    0.0,
                    0.0,  # position
                    0.0,
                    0.0,
                    0.0,  # velocity
                    0.0,
                    0.0,
                    0.0,
                    1.0,  # quaternion (headed backwards 0,0,0,1. default is 1,0,0,0)
                    0.0,
                    0.0,
                    0.0,  # angular velocity
                    100,
                    100,
                    400,
                    100,
                    100,
                    300,
                    400,
                    300,  # image features
                ]
            )
        )

        # PBVS parameters
        self.ineq_bounds = 1e4
        self.w_slack = 0
        self.p_obj0 = p_obj if p_obj is not None else np.array([-10.0, 0.0, 0.0])

        # Pixel bounds for image features
        self.s_min = 10
        self.s_max = 640 - 10

        self.Z0 = Z if Z is not None else np.array([1.0] * 4)

        # Initialize cost functions
        self.set_cost_functions()

        # Build the solver
        self.build_solver()

        # Storage for warm starting
        self.optimal_solution = None

    def set_cost_functions(self):
        """
        Create CasADi functions for the MPC cost objective

        The cost function supports hybrid PBVS/IBVS control through dynamic weighting:
        - Robot state errors (position, velocity, quaternion, angular velocity) are weighted by Q
        - Image feature errors are weighted by S
        - Q and S are dynamically adjusted based on w_p (PBVS weight)
        """
        # Create symbolic variables
        Q = ca.MX.sym("Q", self.Nopt-1, self.Nopt-1)
        R = ca.MX.sym("R", self.Nu, self.Nu)
        S = ca.MX.sym("S", 8, 8)  # Feature weights

        x = ca.MX.sym("x", self.Nopt)  # Robot states (no features)
        xr = ca.MX.sym("xr", self.Nopt)
        s = ca.MX.sym("s", 8)  # Features
        sr = ca.MX.sym("sr", 8)
        u = ca.MX.sym("u", self.Nu)
        ur = ca.MX.sym("ur", self.Nu)

        # State error (special handling for quaternion)
        p_err = x[0:3] - xr[0:3]
        v_err = x[3:6] - xr[3:6]
        q_err = self._quaternion_error_symbolic(x[6:10], xr[6:10])
        w_err = x[10:13] - xr[10:13]

        e_x = ca.vertcat(p_err, v_err, q_err, w_err)
        # e_s = s - sr
        e_u = u - ur

        # Running cost
        ln = (
            ca.mtimes([e_x.T, Q, e_x])
            # + ca.mtimes([e_s.T, S, e_s])
            + ca.mtimes([e_u.T, R, e_u])
        )
        self.running_cost = ca.Function("ln", [x, xr, s, sr, Q, S, u, ur, R], [ln])

        # Terminal cost (using same structure but with different weights)
        Q_e = ca.MX.sym("Q_e", self.Nopt-1, self.Nopt-1)
        S_e = ca.MX.sym("S_e", 8, 8)

        # lN = ca.mtimes([e_x.T, Q_e, e_x]) + ca.mtimes([e_s.T, S_e, e_s])
        lN = ca.mtimes([e_x.T, Q_e, e_x])
        self.terminal_cost = ca.Function("lN", [x, xr, s, sr, Q_e, S_e], [lN])

    def _quaternion_error_symbolic(self, q, q_ref):
        """Compute quaternion error for symbolic CasADi variables"""
        # q_err = q_ref^(-1) * q
        # For unit quaternions: q^(-1) = [qw, -qx, -qy, -qz]
        q_ref_inv = ca.vertcat(q_ref[0], -q_ref[1:4])

        # Quaternion multiplication
        w1, x1, y1, z1 = q_ref_inv[0], q_ref_inv[1], q_ref_inv[2], q_ref_inv[3]
        w2, x2, y2, z2 = q[0], q[1], q[2], q[3]

        q_err = ca.vertcat(
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        )

        return q_err[1:4]  # Return vector part

    def build_solver(self):
        """Build the NLP solver"""
        build_solver_start = time.time()

        # Define weight matrices
        # Position-based VS weights
        Qp = ca.diag(
            ca.vertcat(
                ca.DM.ones(3) * self.Qp_p,  # Position
                ca.DM.ones(3) * 1e3,  # Velocity
                ca.DM.ones(3) * self.Qp_q,  # Quaternion (vector part)
                ca.DM.ones(3) * 4e2,  # Angular velocity
            )
        )

        # Image-based VS weights
        Qs = ca.diag(
            ca.vertcat(
                ca.DM.zeros(3),  # Position
                ca.DM.ones(3) * 65e2,  # Velocity
                ca.DM.zeros(3),  # Quaternion
                ca.DM.ones(3) * 2e3,  # Angular velocity
            )
        )

        S_s = ca.diag(ca.DM.ones(8) * self.w_features)
        R = ca.diag(ca.DM.ones(4) * 1e1)

        

        # Create optimization variables using struct
        opt_var = ctools.struct_symMX(
            [
                ctools.entry("u", shape=(self.Nu,), repeat=self.N),
                ctools.entry("x", shape=(self.Nx,), repeat=self.N + 1),
                ctools.entry("slack", shape=(1,), repeat=self.N + 1),
            ]
        )
        self.opt_var = opt_var
        self.num_var = opt_var.size

        # Create parameter struct
        x0 = ca.MX.sym("x0", self.Nx)
        x_ref = ca.reshape(ca.MX.sym("x_ref", self.Nx, self.N + 1), (-1, 1))
        u_ref = ca.reshape(ca.MX.sym("u_ref", self.Nu, self.N), (-1, 1))
        p_obj = ca.MX.sym("p_obj", 3)
        Z = ca.MX.sym("Z", 4)
        w_p = ca.MX.sym("w_p", 1)  # PBVS weight (w_s = 1 - w_p for IBVS)
        w_slack = ca.MX.sym("w_slack", 1)  # Penalty weight for constraint violation

        param_s = ca.vertcat(x0, x_ref, u_ref, p_obj, Z, w_p, w_slack)

        # Initialize constraints and cost
        obj = ca.MX(0)
        con_eq = []
        con_ineq = []
        con_ineq_lb = []
        con_ineq_ub = []

        # Initial condition constraint
        con_eq.append(opt_var["x", 0] - x0)

        # Decision variable bounds
        self.optvar_lb = opt_var(-ca.inf)
        self.optvar_ub = opt_var(ca.inf)

        # Set control bounds
        for t in range(self.N):
            self.optvar_lb["u", t] = ca.DM.zeros(self.Nu)
            self.optvar_ub["u", t] = ca.DM.ones(self.Nu) * self.model.max_thrust

        # Dynamic weights based on Lyapunov
        # w_p = PBVS weight, w_s = IBVS weight (w_s = 1 - w_p)
        # Q = w_p * Qp + (1 - w_p) * Qs  # Robot state weights
        Q = Qp
        S = (1 - w_p) * S_s  # Image feature weights

        # Terminal weights
        Q_e = 20 * Q  # Will be modified by w_p
        S_e = 100 * S  # Will be modified by w_p

        # Generate MPC Problem
        for t in range(self.N):
            # Get variables
            x_t = opt_var["x", t]
            u_t = opt_var["u", t]
            # slack_t = opt_var["slack", t]

            # Get references
            x_r = x_ref[t * self.Nx : (t + 1) * self.Nx]
            u_r = ca.DM.zeros(self.Nu)
            

            # Extract states
            robot_states = x_t[0 : self.Nopt]
            features = x_t[self.Nopt : self.Nx]
            robot_ref = x_r[0 : self.Nopt]
            feature_ref = x_r[self.Nopt : self.Nx]

            # Dynamics constraint
            params = ca.vertcat(p_obj, Z)
            x_t_next = self.model.f_rk4(
                x_t, u_t, params)
            con_eq.append(x_t_next - opt_var["x", t + 1])

            # Visual bearing constraint with slack
            # g_k = self.model.f_constraint(x_t, params)
            # con_ineq.append(g_k - slack_t)
            # con_ineq_ub.append(0)
            # con_ineq_lb.append(-self.ineq_bounds)

            # Slack variable bounds
            # con_ineq.append(slack_t)
            # con_ineq_ub.append(ca.inf)
            # con_ineq_lb.append(0)

            # Velocity constraints
            v_k = x_t[3:6]
            w_k = x_t[10:13]
            con_ineq.append(ca.vertcat(v_k, w_k))
            con_ineq_ub.append(ca.DM.ones(6) * self.vel_limit)
            con_ineq_lb.append(-ca.DM.ones(6) * self.vel_limit)

            # Cost function
            # The total cost blends PBVS and IBVS based on w_p:
            # - When w_p = 1: Pure PBVS (position tracking)
            # - When w_p = 0: Pure IBVS (image feature tracking)
            # - 0 < w_p < 1: Hybrid mode
            obj += (
                self.running_cost(
                    robot_states, robot_ref, features, feature_ref, Q, S, u_t, u_r, R
                )
            )
            # obj += w_slack * slack_t**2  # Penalty for constraint violation

        # Terminal constraints and cost
        x_N = opt_var["x", self.N]
        slack_N = opt_var["slack", self.N]
        x_r_N = x_ref[self.N * self.Nx : (self.N + 1) * self.Nx]

        # Extract terminal states
        robot_states_N = x_N[0 : self.Nopt]
        features_N = x_N[self.Nopt : self.Nx]
        robot_ref_N = x_r_N[0 : self.Nopt]
        feature_ref_N = x_r_N[self.Nopt : self.Nx]

        # Terminal bearing constraint
        params = ca.vertcat(p_obj, Z)
        # g_N = self.model.f_constraint(x_N, params)
        # con_ineq.append(g_N - slack_N)
        # con_ineq_ub.append(0)
        # con_ineq_lb.append(-self.ineq_bounds)

        # # Terminal slack bounds
        # con_ineq.append(slack_N)
        # con_ineq_ub.append(ca.inf)
        # con_ineq_lb.append(0)

        # Terminal cost
        # Dynamic terminal weights based on PBVS/IBVS mode

        obj += self.terminal_cost(
            robot_states_N, robot_ref_N, features_N, feature_ref_N, Q_e, S_e
        )
        # obj += w_slack * slack_N**2

        # Combine all constraints
        num_eq_con = ca.vertcat(*con_eq).size1()
        num_ineq_con = ca.vertcat(*con_ineq).size1()

        # Equality constraints bounds (g(x) = 0)
        con_eq_lb = np.zeros((num_eq_con,))
        con_eq_ub = np.zeros((num_eq_con,))

        # Set all constraints
        con = ca.vertcat(*(con_eq + con_ineq))
        self.con_lb = ca.vertcat(con_eq_lb, *con_ineq_lb)
        self.con_ub = ca.vertcat(con_eq_ub, *con_ineq_ub)

        # Build NLP Solver
        nlp = dict(x=opt_var, f=obj, g=con, p=param_s)
        options = {
            "ipopt.print_level": 0,
            "ipopt.max_iter": 100,
            "ipopt.tol": 1e-4,
            "ipopt.warm_start_init_point": "yes",
            # "ipopt.warm_start_bound_push": 1e-6,
            # 'ipopt.check_derivatives_for_naninf': "yes",
            "print_time": False,
            "verbose": True,
            "expand": True,
        }

        self.solver = ca.nlpsol("spacecraft_vs_mpc_cs", "ipopt", nlp, options)

        print("\n________________________________________")
        print(f"# Time to build MPC solver: {time.time() - build_solver_start:.3f} sec")
        print(f"# Number of variables: {self.num_var}")
        print(f"# Number of equality constraints: {num_eq_con}")
        print(f"# Number of inequality constraints: {num_ineq_con}")
        print(f"# Horizon: {self.N} steps ({self.Tf:.1f}s)")
        print("----------------------------------------\n")

    def define_lyapunov_weight(self, x, x_ref, s_dot):
        """
        Compute Lyapunov-based weights for hybrid PBVS/IBVS control

        Returns:
            w_p: Weight for PBVS (Position-Based Visual Servoing)
            w_s: Weight for IBVS (Image-Based Visual Servoing)
            Note: w_p + w_s = 1
        """
        e_p = x[0:3] - x_ref[0:3]
        v = x[3:6]
        e_s = x[13:21] - x_ref[13:21]

        Qp_V = ca.diag(ca.DM.ones(3) * self.Qp_p)
        S = ca.diag(ca.DM.ones(8) * self.w_features * 30)

        Vp_dot = ca.dot(e_p, ca.mtimes(Qp_V, v))
        Vs_dot = ca.dot(e_s, ca.mtimes(S, s_dot))

        k = 3.5  # Sharpness of softmax

        softmax_p = ca.exp(-k * Vp_dot)
        softmax_p = ca.if_else(Vp_dot > 0.02, 0, softmax_p)
        softmax_s = ca.exp(-k * Vs_dot)
        eps = 1e-5

        w_s = softmax_s / (softmax_p + softmax_s + eps)  # IBVS weight
        w_p = 1.0 - w_s  # PBVS weight

        return (
            float(w_p),
            float(w_s),
            float(Vp_dot),
            float(Vs_dot),
            float(softmax_p),
            float(softmax_s),
        )

    def update_constraints(self, servoing_enabled):
        """
        Update the slack penalty weight based on servoing mode

        Args:
            servoing_enabled: If True, enables soft constraints with slack penalty
        """
        self.w_slack = (
            2e3 if servoing_enabled else 0
        )  # Penalty for constraint violation

    def solve(self, x0, verbose=False, ref=None, p_obj=None, Z=None, hybrid_mode=False):
        """
        Solve the MPC problem

        Args:
            x0: Current state
            verbose: Print debug information
            ref: Reference trajectory (nx+nu) x (N+1)
            p_obj: Object position
            Z: Feature depths
            hybrid_mode: Enable hybrid PBVS/IBVS mode

        Returns:
            simU: Control trajectory
            simX: State trajectory
            w_p: PBVS weight (Position-Based Visual Servoing)
            w_s: IBVS weight (Image-Based Visual Servoing)
        """
        solver_start = time.time()

        # Set defaults
        if ref is None:
            ref = np.zeros((self.model.nx + self.model.nu, self.N + 1))
            ref[6, :] = 1.0  # Identity quaternion

        p_obj = p_obj if p_obj is not None else self.p_obj0
        Z = Z if Z is not None else self.Z0

        # Extract references
        x_ref = ref[: self.model.nx, :]
        u_ref = ref[self.model.nx :, : self.N]

        # Compute Lyapunov weights for PBVS/IBVS hybrid control
        if hybrid_mode and not self.ibvs_mode:
            # Compute feature dynamics for weight calculation
            L_val = self.model.f_interaction(x0[13:21], Z)
            s_dot = self.model.f_feature_dynamics(L_val, x0[3:6], x0[10:13])
            s_dot = np.array(s_dot).flatten()

            w_p, w_s, Vp_dot, Vs_dot, softmax_p, softmax_s = (
                self.define_lyapunov_weight(x0, x_ref[:, 0], s_dot)
            )

            if w_p < 0.05:  # Switch to IBVS when PBVS weight is low
                self.ibvs_mode = True
        elif hybrid_mode and self.ibvs_mode:
            w_p = 0.0  # Pure IBVS mode
            w_s = 1.0
            Vp_dot = Vs_dot = softmax_p = softmax_s = 0.0
        else:
            self.ibvs_mode = False
            w_p = 1.0  # Pure PBVS mode
            w_s = 0.0
            Vp_dot = Vs_dot = softmax_p = softmax_s = 0.0

        # Initialize warm start
        if self.optimal_solution is not None:
            # Shift previous solution for warm start
            self.optvar_init = self.opt_var(0)
            self.optvar_init["x"] = self.optimal_solution["x"][1:] + [
                ca.DM.zeros(self.Nx)
            ]
            self.optvar_init["u"] = self.optimal_solution["u"][1:] + [
                ca.DM.zeros(self.Nu)
            ]
            self.optvar_init["slack"] = self.optimal_solution["slack"][1:] + [ca.DM(0)]
        else:
            # Initialize with zeros
            self.optvar_init = self.opt_var(0)

        # Override initial state
        self.optvar_init["x", 0] = x0

        # Prepare parameters
        x_ref_flat = x_ref.reshape(-1, 1, order="F")
        u_ref_flat = u_ref.reshape(-1, 1, order="F")
        param = ca.vertcat(x0, x_ref_flat, u_ref_flat, p_obj, Z, w_p, self.w_slack)

        # Solver arguments
        args = dict(
            x0=self.optvar_init,
            lbx=self.optvar_lb,
            ubx=self.optvar_ub,
            lbg=self.con_lb,
            ubg=self.con_ub,
            p=param,
        )

        # Solve NLP
        sol = self.solver(**args)
        status = self.solver.stats()["return_status"]
        optvar = self.opt_var(sol["x"])
        self.optimal_solution = optvar

        solve_time = time.time() - solver_start

        if verbose:
            print(
                f"MPC - CPU time: {solve_time*1000:.2f} ms | Cost: {float(sol['f']):9.2f} | Status: {status}"
            )
            if hybrid_mode:
                print(f"===== Lyapunov Values =====")
                print(f"Vp_dot: {Vp_dot:.2f}, Vs_dot: {Vs_dot:.2f}")
                print(f"softmax_p: {softmax_p:.2f}, softmax_s: {softmax_s:.2f}")
                print(f"w_p (PBVS): {w_p:.2f}, w_s (IBVS): {w_s:.2f}")
                print(f"Mode: {'IBVS' if self.ibvs_mode else 'PBVS/Hybrid'}")

        # Extract solution
        simX = np.zeros((self.N + 1, self.Nx))
        simU = np.zeros((self.N, self.Nu))

        for i in range(self.N + 1):
            simX[i, :] = np.array(optvar["x", i]).flatten()
        for i in range(self.N):
            simU[i, :] = np.array(optvar["u", i]).flatten()

        #debug: display simulated state of the pose
        if verbose:
            print("\n===== Simulated States =====")
            for i in range(self.N + 1):
                print(f"t={i*self.model.dt:.2f}s: {simX[i, :3]} (pos)")

        return simU, simX, w_p, w_s

    def reset_warm_start(self):
        """Reset the warm start information"""
        self.optimal_solution = None
        self.ibvs_mode = False
