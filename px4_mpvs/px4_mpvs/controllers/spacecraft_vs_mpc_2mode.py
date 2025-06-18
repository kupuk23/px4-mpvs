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
import scipy.linalg
import casadi as cs
import time


class SpacecraftVSMPC:
    def __init__(self, model, p_obj=None, x0=None, Z=None, mode="pbvs"):
        # BEARING PENALTY Variables
        self.mode = mode

        self.vel_limit = 5  # np.inf .1
        self.model = model
        self.Tf = 5.0
        self.N = 49

        self.x0 = (
            x0
            if x0 is not None
            else np.array(
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            )
        )
        if self.mode == "pbvs":
            self.ineq_bounds = 1e4
            self.w_slack = 0
            self.p_obj0 = p_obj if p_obj is not None else np.array([10.0, 0.0, 0.0])
            self.ocp_solver, self.integrator = self.setup(
                self.x0, self.N, self.Tf, self.p_obj0
            )
        elif self.mode == "ibvs":
            assert Z is not None, "Z must be provided for IBVS mode"
            # pixel bounds for image features
            self.s_min = 10
            self.s_max = 640 - 10 # 640x480 image

            self.Z0 = Z if Z is not None else np.array([1.0] * 4)
            self.ocp_solver, self.integrator = self.setup(
                self.x0, self.N, self.Tf, Z0=self.Z0
            )

    def set_bearing_constraints(self, ocp):
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
        return ocp

    def setup(self, x0, N_horizon, Tf, p_obj0=None, Z0=None):
        # create ocp object to formulate the OCP
        ocp = AcadosOcp()

        # set model
        model = self.model.get_acados_model()
        Fmax = self.model.max_thrust

        ocp.model = model

        nx = model.x.size()[0]
        nu = model.u.size()[0]
        print(f"nx: {nx}, nu: {nu}")

        # set dimensions
        ocp.dims.N = N_horizon
        ocp.solver_options.N_horizon = N_horizon

        # set cost
        Q_mat = np.diag(
            [
                *[5e1] * 3,  # Position weights (x, y, z) # 5e1
                *[1e1] * 3,  # Velocity weights (vx, vy, vz) 
                8e3,  # Quaternion scalar part, 8e3 default
                *[1e1] * 3,  # angular vel part, (ωx, ωy, ωz)
            ]
        )
        Q_e = 20 * Q_mat
        R_mat = np.diag([1e1] * 4)

        # References:
        x_ref = cs.MX.sym("x_ref", (nx, 1))  # 13 states + 8 features
        u_ref = cs.MX.sym("u_ref", (nu, 1))

        # Calculate errors
        # x : p,v,q,w               , R9 x SO(3)
        # u : Fx,Fy,Fz,Mx,My,Mz     , R6

        x = ocp.model.x
        u = ocp.model.u

        x_error = x[0:3] - x_ref[0:3]
        x_error = cs.vertcat(x_error, x[3:6] - x_ref[3:6])
        x_error = cs.vertcat(x_error, 1 - (x[6:10].T @ x_ref[6:10]) ** 2)
        x_error = cs.vertcat(x_error, x[10:13] - x_ref[10:13])

        u_error = u - u_ref

        # define cost with parametric reference
        ocp.cost.cost_type = "EXTERNAL"
        ocp.cost.cost_type_e = "EXTERNAL"

        if self.mode == "pbvs":
            assert p_obj0 is not None, "p_obj0 must be provided for PBVS mode"
            # Initialize parameters
            p_obj = ocp.model.p[0:3]

            ocp.model.p = cs.vertcat(x_ref, u_ref, p_obj)

            p_0 = np.concatenate(
                (x0, np.zeros(nu), p_obj0)
            )  # First step is error 0 since x_ref = x0

            ocp = self.set_bearing_constraints(ocp)

        elif self.mode == "ibvs":
            assert Z0 is not None, "Z must be provided for IBVS mode"

            x0 = np.concatenate((x0, np.zeros(8)))
            # add x_error for image features
            x_error = cs.vertcat(x_error, x[13:] - x_ref[13:])

            Z = ocp.model.p
            ocp.model.p = cs.vertcat(x_ref, u_ref, Z)

            p_0 = np.concatenate(
                (x0, np.zeros(nu), Z0)
            )  # s0 = features state, Z = feature depth

            # set p and q weight to 0
            Q_mat = np.diag(
                [
                    *[0] * 3,  # Position weights (x, y, z)
                    *[6e3] * 3,  # Velocity weights (vx, vy, vz) # 5e1
                    0,  # Quaternion scalar part, 8e3 default
                    *[5e2] * 3,  # angular vel (ωx, ωy, ωz) # 5e1
                    *[7e-3] * 8,  # Image feature weights
                ]
            )
            Q_e = 20 * Q_mat
            Q_e[9:,9:] = 50 * Q_mat[9:,9:]

            # set bounds for image features (x coordinates)
            ocp.constraints.idxbx = np.array(
                [13, 15, 17, 19]
            )
            ocp.constraints.lbx = np.array([self.s_min] * 4)
            ocp.constraints.ubx = np.array([self.s_max] * 4)

        ocp.model.cost_expr_ext_cost = (
            x_error.T @ Q_mat @ x_error + u_error.T @ R_mat @ u_error
        )

        ocp.model.cost_expr_ext_cost_e = x_error.T @ Q_e @ x_error

        ocp.parameter_values = p_0

        # set constraints
        ocp.constraints.lbu = np.array([-Fmax, -Fmax, -Fmax, -Fmax])
        ocp.constraints.ubu = np.array([+Fmax, +Fmax, +Fmax, +Fmax])
        ocp.constraints.idxbu = np.array([0, 1, 2, 3])
        ocp.constraints.x0 = x0

        # set bounds for velocity

        ocp.constraints.idxbx = np.array(
            [3, 4, 5, 10, 11, 12]
        )  # vx, vy, vz, ωx, ωy, ωz
        ocp.constraints.lbx = np.array(
            [-self.vel_limit] * 6
        )
        ocp.constraints.ubx = np.array(
            [self.vel_limit] * 6
        )

        

        # set options
        ocp.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM"
        # PARTIAL_CONDENSING_HPIPM, FULL_CONDENSING_QPOASES, FULL_CONDENSING_HPIPM,
        # PARTIAL_CONDENSING_QPDUNES, PARTIAL_CONDENSING_OSQP, FULL_CONDENSING_DAQP
        ocp.solver_options.hessian_approx = "GAUSS_NEWTON"  # 'GAUSS_NEWTON', 'EXACT'
        ocp.solver_options.integrator_type = "ERK"

        # ocp.solver_options.print_level = 1

        use_RTI = True
        if use_RTI:
            ocp.solver_options.nlp_solver_type = "SQP_RTI"  # SQP_RTI, SQP
            ocp.solver_options.sim_method_num_stages = 4
            ocp.solver_options.sim_method_num_steps = 3
        else:
            ocp.solver_options.nlp_solver_type = "SQP"  # SQP_RTI, SQP

        # ocp.solver_options.qp_solver_cond_N = N_horizon
        # ocp.solver_options.print_level = 6

        # set prediction horizon
        ocp.solver_options.tf = Tf

        ocp_solver = AcadosOcpSolver(ocp, json_file="acados_ocp.json")
        # create an integrator with the same settings as used in the OCP solver.
        acados_integrator = AcadosSimSolver(ocp, json_file="acados_ocp.json")

        return ocp_solver, acados_integrator

    def update_constraints(self, servoing_enabled):
        # Update the constraints based on the servoing_enabled flag
        self.w_slack = 5e2 if servoing_enabled else 0  # 4e2

    def solve(self, x0, verbose=False, ref=None, p_obj=None, Z=None):
        if self.mode == "ibvs":
            assert Z is not None, "Z must be provided for IBVS mode"

        elif self.mode == "pbvs":
            p_obj = p_obj if p_obj is not None else self.p_obj0

        # preparation phase
        ocp_solver = self.ocp_solver

        # Set reference, create zero reference
        if ref is None:
            zero_ref = np.zeros(
                self.model.get_acados_model().x.size()[0]
                + self.model.get_acados_model().u.size()[0]
            )
            zero_ref[6] = 1.0

        for i in range(self.N + 1):
            if i != self.N and i != 0:
                if self.mode == "pbvs":
                    self.ocp_solver.cost_set(i, "Zu", np.array([self.w_slack]))
                    self.ocp_solver.cost_set(i, "zu", np.array([self.w_slack]))

            if ref is not None:
                # Assumed ref structure: (nx+nu) x N+1
                # NOTE: last u_ref is not used
                if self.mode == "pbvs":
                    p_i = np.concatenate([ref[:, i], p_obj])
                else:
                    p_i = np.concatenate([ref[:, i], Z])
                ocp_solver.set(i, "p", p_i)
            else:
                # set all references to 0
                if self.mode == "pbvs":
                    p_i = np.concatenate([zero_ref, p_obj])
                else:
                    p_i = np.concatenate([zero_ref, Z])
                ocp_solver.set(i, "p", p_i)

        # self.ocp_solver.cost_set(self.N, "zu_e", np.array([self.w_slack]))
        # self.ocp_solver.cost_set(self.N, "Zu_e", np.array([self.w_slack]))

        # set initial state
        ocp_solver.set(0, "lbx", x0.flatten())
        ocp_solver.set(0, "ubx", x0.flatten())

        status = ocp_solver.solve()
        if verbose:
            self.ocp_solver.print_statistics()  # encapsulates: stat = ocp_solver.get_stats("statistics")
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

        return simU, simX
