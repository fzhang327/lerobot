# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Disturbance observer-based admittance controller middleware.

This module provides :class:`ESOAdmittanceController`, a light-weight middleware
layer intended to sit between a low-rate policy (e.g. VLA) and a high-rate joint
servo loop. It combines:

1. A linear extended state observer (LESO) to estimate total disturbances.
2. A force estimate derived from the observer disturbance state.
3. A virtual mass-spring-damper admittance model to generate compliant,
   smooth joint commands.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import numpy.typing as npt

ArrayLike = npt.NDArray[np.float64]


class ESOAdmittanceController:
    """Disturbance Observer-based Admittance Controller for joint space control.

    The controller tracks a target joint trajectory while allowing compliant
    response to estimated external disturbances, without requiring a force
    sensor.

    Notes:
        - LESO equations (continuous-time) are discretized with explicit Euler.
        - The previous commanded position is used as the observer input ``u``.
        - ``D_d`` is set to critical damping by default:
          ``D_d = 2 * sqrt(M_d * K_d)``.
    """

    def __init__(
        self,
        dof: int,
        *,
        b0: float | npt.ArrayLike,
        omega_o: float,
        k_d: float | npt.ArrayLike,
        m_d: float | npt.ArrayLike = 1.0,
        d_d: float | npt.ArrayLike | None = None,
        force_deadband: float = 0.02,
        force_limit: float = 8.0,
        target_cutoff_hz: float | None = 3.0,
        gravity_fn: Callable[[ArrayLike], ArrayLike] | None = None,
    ) -> None:
        """Initialize controller parameters and internal states.

        Args:
            dof: Number of robot joints.
            b0: Physical system gain (scalar or per-joint vector), tunable.
            omega_o: LESO bandwidth parameter.
            k_d: Virtual stiffness ``K_d`` (scalar or per-joint vector), tunable.
            m_d: Virtual inertia ``M_d`` (scalar or per-joint vector). Defaults to 1.
            d_d: Virtual damping ``D_d`` (scalar or per-joint vector).
                If ``None``, uses critical damping.
            force_deadband: Absolute force threshold below which force is set to zero.
            force_limit: Safety force limit. If exceeded, command freezes at current pose.
            target_cutoff_hz: Low-pass cutoff for policy target smoothing.
                If ``None`` or <=0, filtering is disabled.
            gravity_fn: Optional gravity compensation callback ``gravity_fn(q)``.
                If omitted, zero gravity compensation is used.
        """
        self.dof = int(dof)
        if self.dof <= 0:
            msg = "dof must be > 0"
            raise ValueError(msg)

        self.b0 = self._as_joint_array(b0, name="b0")
        if np.any(np.isclose(self.b0, 0.0)):
            msg = "b0 must be non-zero for every joint"
            raise ValueError(msg)

        self.omega_o = float(omega_o)
        if self.omega_o <= 0:
            msg = "omega_o must be > 0"
            raise ValueError(msg)

        self.beta1 = 3.0 * self.omega_o
        self.beta2 = 3.0 * (self.omega_o**2)
        self.beta3 = self.omega_o**3

        self.m_d = self._as_joint_array(m_d, name="m_d")
        if np.any(self.m_d <= 0):
            msg = "m_d must be > 0 for every joint"
            raise ValueError(msg)

        self.k_d = self._as_joint_array(k_d, name="k_d")
        if np.any(self.k_d < 0):
            msg = "k_d must be >= 0 for every joint"
            raise ValueError(msg)

        self.d_d = (
            2.0 * np.sqrt(self.m_d * self.k_d)
            if d_d is None
            else self._as_joint_array(d_d, name="d_d")
        )

        self.force_deadband = float(force_deadband)
        self.force_limit = float(force_limit)
        self.target_cutoff_hz = (
            None if target_cutoff_hz is None or target_cutoff_hz <= 0 else float(target_cutoff_hz)
        )
        self.gravity_fn = gravity_fn

        # LESO states: z1 ~ position, z2 ~ velocity, z3 ~ total disturbance.
        self.z1 = np.zeros(self.dof, dtype=np.float64)
        self.z2 = np.zeros(self.dof, dtype=np.float64)
        self.z3 = np.zeros(self.dof, dtype=np.float64)

        # Admittance states.
        self.q_c = np.zeros(self.dof, dtype=np.float64)
        self.dq_c = np.zeros(self.dof, dtype=np.float64)

        # Target smoothing state and observer input memory.
        self.q_vla_filt = np.zeros(self.dof, dtype=np.float64)
        self.u_prev = np.zeros(self.dof, dtype=np.float64)
        self.initialized = False

    def reset(self, q_init: npt.ArrayLike) -> None:
        """Reset observer and controller states at a given joint position."""
        q0 = self._as_joint_array(q_init, name="q_init")
        self.z1 = q0.copy()
        self.z2.fill(0.0)
        self.z3.fill(0.0)

        self.q_c = q0.copy()
        self.dq_c.fill(0.0)
        self.q_vla_filt = q0.copy()
        self.u_prev = q0.copy()
        self.initialized = True

    def gravity_compensation(self, q: ArrayLike) -> ArrayLike:
        """Return gravity torque estimate for joint position ``q``.

        This is intentionally a placeholder interface. Override this method or
        pass ``gravity_fn`` in ``__init__`` for robot-specific dynamics.
        """
        if self.gravity_fn is None:
            return np.zeros_like(q)
        g = np.asarray(self.gravity_fn(q), dtype=np.float64)
        if g.shape != (self.dof,):
            msg = f"gravity_fn must return shape ({self.dof},), got {g.shape}"
            raise ValueError(msg)
        return g

    def update(
        self,
        q_vla_target: npt.ArrayLike,
        q_measured: npt.ArrayLike,
        dt: float,
    ) -> ArrayLike:
        """Advance LESO + admittance dynamics and return compliant command.

        Args:
            q_vla_target: Policy target joint position.
            q_measured: Current measured joint position.
            dt: Controller time step (seconds).

        Returns:
            ``q_compliant_cmd`` to send to servos.
        """
        q_target = self._as_joint_array(q_vla_target, name="q_vla_target")
        q_meas = self._as_joint_array(q_measured, name="q_measured")
        dt = float(dt)
        if dt <= 0:
            msg = "dt must be > 0"
            raise ValueError(msg)

        if not self.initialized:
            self.reset(q_meas)

        q_target_smooth = self._lowpass_target(q_target, dt)

        # LESO (physical observer):
        # e = q_meas - z1
        # z1_dot = z2 + beta1 * e
        # z2_dot = z3 + b0 * u + beta2 * e
        # z3_dot = beta3 * e
        e = q_meas - self.z1
        self.z1 = self.z1 + dt * (self.z2 + self.beta1 * e)
        self.z2 = self.z2 + dt * (self.z3 + self.b0 * self.u_prev + self.beta2 * e)
        self.z3 = self.z3 + dt * (self.beta3 * e)

        # Disturbance-based force estimation.
        tau_ext_hat = (self.z3 / self.b0) - self.gravity_compensation(q_meas)
        tau_ext_hat = self._apply_deadband(tau_ext_hat, self.force_deadband)

        # Safety: freeze motion if force estimate exceeds configured limit.
        if np.any(np.abs(tau_ext_hat) > self.force_limit):
            self.q_c = q_meas.copy()
            self.dq_c.fill(0.0)
            self.u_prev = q_meas.copy()
            return q_meas.copy()

        # Virtual admittance model:
        # M_d qdd_c + D_d qd_c + K_d (q_c - q_vla) = tau_ext_hat
        qdd_c = (tau_ext_hat - self.d_d * self.dq_c - self.k_d * (self.q_c - q_target_smooth)) / self.m_d
        self.dq_c = self.dq_c + dt * qdd_c
        self.q_c = self.q_c + dt * self.dq_c

        self.u_prev = self.q_c.copy()
        return self.q_c.copy()

    def _lowpass_target(self, q_target: ArrayLike, dt: float) -> ArrayLike:
        """First-order low-pass for low-frequency step-like policy targets."""
        if self.target_cutoff_hz is None:
            self.q_vla_filt = q_target.copy()
            return self.q_vla_filt

        tau = 1.0 / (2.0 * np.pi * self.target_cutoff_hz)
        alpha = dt / (tau + dt)
        self.q_vla_filt = self.q_vla_filt + alpha * (q_target - self.q_vla_filt)
        return self.q_vla_filt

    def _as_joint_array(self, value: float | npt.ArrayLike, *, name: str) -> ArrayLike:
        """Convert scalar/array to shape ``(dof,)`` float64 array."""
        arr = np.asarray(value, dtype=np.float64)
        if arr.ndim == 0:
            return np.full(self.dof, float(arr), dtype=np.float64)
        if arr.shape != (self.dof,):
            msg = f"{name} must be scalar or shape ({self.dof},), got {arr.shape}"
            raise ValueError(msg)
        return arr

    @staticmethod
    def _apply_deadband(signal: ArrayLike, deadband: float) -> ArrayLike:
        """Zero-out small signals inside ``[-deadband, deadband]``."""
        if deadband <= 0:
            return signal
        out = signal.copy()
        out[np.abs(out) < deadband] = 0.0
        return out
