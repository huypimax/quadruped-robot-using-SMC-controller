#!/usr/bin/env python3
"""Quadruped simplified dynamics (block-diagonal per leg).

This is a direct port of the Matlab functions you shared:
  - leg_dynamics(q, dq)
  - quadruped_dynamics(q, dq) = blkdiag(leg_dynamics(...))
  - forward_dynamics(q, dq, tau) = inv(M) * (tau - C*dq - G)

Model notes
-----------
- 12 DoF total: FR, FL, RR, RL; each leg is 3 DoF: [hip, thigh, calf].
- The global M and C are block-diagonal (legs are decoupled in this model).
- This is a *simplified* dynamics model intended for control design/SMC.
"""

from __future__ import annotations

import numpy as np


def leg_dynamics(q: np.ndarray, dq: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
	"""Return (M, C, G) for a single 3-DoF leg.

	Args:
		q:  (3,) joint angles [q1, q2, q3]
		dq: (3,) joint velocities

	Returns:
		M: (3,3) inertia matrix
		C: (3,3) coriolis/centrifugal matrix such that tau_c = C @ dq
		G: (3,) gravity vector
	"""

	q = np.asarray(q, dtype=float).reshape(3)
	dq = np.asarray(dq, dtype=float).reshape(3)

	q1, q2, q3 = q
	dq1, dq2, dq3 = dq

	# --- Constants (ported from Matlab) ---
	Ihip_urdf = 0.000469246
	# Equivalent hip inertia around the shoulder roll axis.
	Ihip = Ihip_urdf + (1.013 + 0.166 + 0.060) * (0.08505**2)

	m2 = 1.013
	m3 = 0.166 + 0.06

	l1 = 0.2
	lc1 = 0.1
	lc2 = 0.1

	I2 = 0.005139339
	I3 = 0.003014022 + 9.6e-6

	g = 9.81

	# --- Inertia matrix terms ---
	c3 = np.cos(q3)
	s3 = np.sin(q3)

	M11 = I2 + I3 + m2 * (lc1**2) + m3 * (l1**2 + lc2**2 + 2.0 * l1 * lc2 * c3)
	M12 = I3 + m3 * (lc2**2 + l1 * lc2 * c3)
	M22 = I3 + m3 * (lc2**2)

	M = np.array(
		[
			[Ihip, 0.0, 0.0],
			[0.0, M11, M12],
			[0.0, M12, M22],
		],
		dtype=float,
	)

	# --- Coriolis/Centrifugal matrix ---
	h = -m3 * l1 * lc2 * s3
	C = np.array(
		[
			[0.0, 0.0, 0.0],
			[0.0, h * dq3, h * (dq2 + dq3)],
			[0.0, -h * dq2, 0.0],
		],
		dtype=float,
	)

	# --- Gravity vector ---
	# q1 gravity neglected in this simplified model
	G1 = 0.0
	G2 = (m2 * lc1 + m3 * l1) * g * np.sin(q2) + m3 * lc2 * g * np.sin(q2 + q3)
	G3 = m3 * lc2 * g * np.sin(q2 + q3)
	G = np.array([G1, G2, G3], dtype=float)

	return M, C, G


def quadruped_dynamics(
	q: np.ndarray, dq: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
	"""Return (M, C, G) for the full 12-DoF quadruped (block-diagonal by leg)."""

	q = np.asarray(q, dtype=float).reshape(12)
	dq = np.asarray(dq, dtype=float).reshape(12)

	qFR, qFL, qRR, qRL = q[0:3], q[3:6], q[6:9], q[9:12]
	dqFR, dqFL, dqRR, dqRL = dq[0:3], dq[3:6], dq[6:9], dq[9:12]

	MFR, CFR, GFR = leg_dynamics(qFR, dqFR)
	MFL, CFL, GFL = leg_dynamics(qFL, dqFL)
	MRR, CRR, GRR = leg_dynamics(qRR, dqRR)
	MRL, CRL, GRL = leg_dynamics(qRL, dqRL)

	# Assemble block-diagonal matrices without SciPy
	M = np.zeros((12, 12), dtype=float)
	C = np.zeros((12, 12), dtype=float)

	blocks_M = (MFR, MFL, MRR, MRL)
	blocks_C = (CFR, CFL, CRR, CRL)
	for i in range(4):
		r = slice(3 * i, 3 * (i + 1))
		M[r, r] = blocks_M[i]
		C[r, r] = blocks_C[i]

	G = np.concatenate([GFR, GFL, GRR, GRL]).astype(float)
	return M, C, G


def forward_dynamics(q: np.ndarray, dq: np.ndarray, tau: np.ndarray) -> np.ndarray:
	"""Compute joint accelerations ddq from (q, dq, tau).

	Equivalent to Matlab:
		ddq = M \ (tau - C*dq - G)
	"""
	q = np.asarray(q, dtype=float).reshape(12)
	dq = np.asarray(dq, dtype=float).reshape(12)
	tau = np.asarray(tau, dtype=float).reshape(12)

	M, C, G = quadruped_dynamics(q, dq)
	rhs = tau - (C @ dq) - G
	ddq = np.linalg.solve(M, rhs)
	return ddq


def tau_limit_vector() -> np.ndarray:
	"""Return the per-joint torque limits used in your Matlab init (shape (12,))."""
	tau_max_leg = np.array([20.0, 55.0, 55.0], dtype=float)
	return np.tile(tau_max_leg, 4)


def clip_tau(tau: np.ndarray, tau_max: np.ndarray | None = None) -> np.ndarray:
	"""Elementwise clip of torque."""
	tau = np.asarray(tau, dtype=float).reshape(12)
	if tau_max is None:
		tau_max = tau_limit_vector()
	tau_max = np.asarray(tau_max, dtype=float).reshape(12)
	return np.clip(tau, -tau_max, tau_max)

