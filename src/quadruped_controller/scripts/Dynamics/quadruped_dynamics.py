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


def rpy_to_rotm_safe(rpy: np.ndarray) -> np.ndarray:
	"""Convert roll-pitch-yaw to rotation matrix (Z-Y-X), with safety clamps.

	Args:
		rpy: (3,) [roll, pitch, yaw]

	Returns:
		R: (3,3) rotation from body->world
	"""
	rpy = np.asarray(rpy, dtype=float).reshape(3)
	roll, pitch, yaw = rpy
	# clamp to reasonable ranges (as in MATLAB port)
	roll = float(np.clip(roll, -np.pi, np.pi))
	pitch = float(np.clip(pitch, -np.pi / 2.0, np.pi / 2.0))
	yaw = float(np.clip(yaw, -np.pi, np.pi))
	cr = np.cos(roll)
	sr = np.sin(roll)
	cp = np.cos(pitch)
	sp = np.sin(pitch)
	cy = np.cos(yaw)
	sy = np.sin(yaw)
	Rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]], dtype=float)
	Ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]], dtype=float)
	Rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]], dtype=float)
	return Rz @ Ry @ Rx


def foot_kinematics_A1(q_leg: np.ndarray, side: float, hip_side: float, l1: float, l2: float) -> np.ndarray:
	"""Foot position relative to hip expressed in body/hip frame.

	Args:
		q_leg: (3,) [q1 hip, q2 thigh, q3 calf]
		side: +/-1 (left/right)
	"""
	q = np.asarray(q_leg, dtype=float).reshape(3)
	q1, q2, q3 = q
	d = side * hip_side
	q23 = q2 + q3
	A = l1 * np.cos(q2) + l2 * np.cos(q23)
	B = l1 * np.sin(q2) + l2 * np.sin(q23)
	x = -B
	y = d * np.cos(q1) + A * np.sin(q1)
	z = d * np.sin(q1) - A * np.cos(q1)
	return np.array([x, y, z], dtype=float)


def leg_jacobian_A1(q_leg: np.ndarray, side: float, hip_side: float, l1: float, l2: float) -> np.ndarray:
	"""Analytical Jacobian consistent with foot_kinematics_A1.

	Returns J (3x3) mapping dq -> foot velocity in body frame.
	"""
	q = np.asarray(q_leg, dtype=float).reshape(3)
	q1, q2, q3 = q
	d = side * hip_side
	q23 = q2 + q3
	A = l1 * np.cos(q2) + l2 * np.cos(q23)
	B = l1 * np.sin(q2) + l2 * np.sin(q23)
	dA_dq2 = -B
	dA_dq3 = -l2 * np.sin(q23)
	dB_dq2 = A
	dB_dq3 = l2 * np.cos(q23)
	# x = -B
	dx_dq1 = 0.0
	dx_dq2 = -dB_dq2
	dx_dq3 = -dB_dq3
	# y = d*cos(q1) + A*sin(q1)
	dy_dq1 = -d * np.sin(q1) + A * np.cos(q1)
	dy_dq2 = dA_dq2 * np.sin(q1)
	dy_dq3 = dA_dq3 * np.sin(q1)
	# z = d*sin(q1) - A*cos(q1)
	dz_dq1 = d * np.cos(q1) + A * np.sin(q1)
	dz_dq2 = -dA_dq2 * np.cos(q1)
	dz_dq3 = -dA_dq3 * np.cos(q1)
	J = np.array(
		[
			[dx_dq1, dx_dq2, dx_dq3],
			[dy_dq1, dy_dq2, dy_dq3],
			[dz_dq1, dz_dq2, dz_dq3],
		],
		dtype=float,
	)
	return J


def fcn(
	q: np.ndarray,
	dq: np.ndarray,
	p_body: np.ndarray,
	v_body: np.ndarray,
	rpy_body: np.ndarray,
	omega_body: np.ndarray,
	tau: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
	"""Full dynamics function ported from the Matlab code.

	Signature matches your Matlab function:
	ddq, ddp_body, domega_body, foot_z, contact_Fz = fcn(q, dq, p_body, v_body, rpy_body, omega_body, tau)

	Notes:
	- q, dq: (12,) joint states [FR(3), FL(3), RR(3), RL(3)]
	- p_body, v_body: (3,) body position/velocity in world frame
	- rpy_body: (3,) roll/pitch/yaw
	- omega_body: (3,) angular velocity expressed in body frame
	- tau: (12,) actuator torques

	Returns:
		ddq: (12,) joint accelerations
		ddp_body: (3,) body linear acceleration in world frame
		domega_body: (3,) angular acceleration in body frame
		foot_z: (4,) foot z positions in world frame
		contact_Fz: (4,) contact vertical forces (world frame, +up)
	"""

	# --- outputs / ensure shapes ---
	q = np.asarray(q, dtype=float).reshape(12)
	dq = np.asarray(dq, dtype=float).reshape(12)
	p_body = np.asarray(p_body, dtype=float).reshape(3)
	v_body = np.asarray(v_body, dtype=float).reshape(3)
	rpy_body = np.asarray(rpy_body, dtype=float).reshape(3)
	omega_body = np.asarray(omega_body, dtype=float).reshape(3)
	tau = np.asarray(tau, dtype=float).reshape(12)

	ddq = np.zeros(12, dtype=float)
	ddp_body = np.zeros(3, dtype=float)
	domega_body = np.zeros(3, dtype=float)
	foot_z = np.zeros(4, dtype=float)
	contact_Fz = np.zeros(4, dtype=float)

	# --- Parameters ---
	m_body = 4.713
	I_body = np.diag([0.01683993, 0.056579028, 0.064713601])
	g = 9.81

	# hip positions in body frame (FR, FL, RR, RL)
	hip_pos_body = np.array(
		[
			[0.183, -0.047, 0.0],
			[0.183, 0.047, 0.0],
			[-0.183, -0.047, 0.0],
			[-0.183, 0.047, 0.0],
		],
		dtype=float,
	)

	side_sign = np.array([-1.0, 1.0, -1.0, 1.0], dtype=float)
	hip_side = 0.08505
	l1 = 0.20
	l2 = 0.20

	# ground contact
	K_ground = 6000.0
	B_ground = 350.0
	Fz_max = 120.0

	# body damping
	linear_damping = 2.0
	D_roll = 0.08
	D_pitch = 0.08
	D_yaw = 1.20
	D_angular = np.diag([D_roll, D_pitch, D_yaw])

	# yaw hold
	yaw_ref = 0.0
	K_yaw_hold = 1.50

	# joint damping
	joint_damping = 0.08

	# rotation body->world
	R = rpy_to_rotm_safe(rpy_body)

	# =====================================================
	# Joint acceleration from leg dynamics (Matlab: ddq = M\(tau_eff - C*dq - G))
	# =====================================================
	M, C, G = quadruped_dynamics(q, dq)
	tau_eff = tau - joint_damping * dq
	ddq = np.linalg.solve(M, tau_eff - (C @ dq) - G)
	# Protect from numerical explosion (Matlab clamp)
	# ddq = np.clip(ddq, -300.0, 300.0)

	# =====================================================
	# Body force and torque (Matlab)
	# =====================================================
	F_total_world = np.array([0.0, 0.0, -m_body * g], dtype=float) - linear_damping * v_body
	T_total_body = -(D_angular @ omega_body)
	# yaw hold spring
	yaw_error = rpy_body[2] - yaw_ref
	T_total_body[2] = T_total_body[2] - (K_yaw_hold * yaw_error)

	for i in range(4):
		qi = q[3 * i : 3 * i + 3]
		dqi = dq[3 * i : 3 * i + 3]
		side = side_sign[i]
		p_foot_body = foot_kinematics_A1(qi, side, hip_side, l1, l2)
		# foot pos in body frame (from body origin)
		r_body = hip_pos_body[i, :] + p_foot_body
		# foot pos in world frame
		p_foot_world = R @ r_body + p_body
		foot_z[i] = p_foot_world[2]
		# foot velocity in world: v_body + R*(omega_body x r_body + J @ dq_leg)
		J_leg = leg_jacobian_A1(qi, side, hip_side, l1, l2)
		v_foot_body = np.cross(omega_body, p_foot_body) + J_leg @ dqi
		v_foot_world = v_body + R @ v_foot_body
		vz = v_foot_world[2]
		# Ground contact (Matlab)
		if p_foot_world[2] < 0.0:
			penetration = -p_foot_world[2]
			Fz = K_ground * penetration - B_ground * vz
			if Fz < 0.0:
				Fz = 0.0
			if Fz > Fz_max:
				Fz = Fz_max
		else:
			Fz = 0.0
		contact_Fz[i] = Fz

		F_contact_world = np.array([0.0, 0.0, Fz], dtype=float)
		F_total_world = F_total_world + F_contact_world

		# Moment in BODY frame
		F_contact_body = R.T @ F_contact_world
		T_total_body = T_total_body + np.cross(r_body, F_contact_body)

	# =====================================================
	# Body linear acceleration (Matlab)
	# =====================================================
	ddp_body = F_total_world / m_body
	ddp_body = np.clip(ddp_body, -80.0, 80.0)

	# =====================================================
	# Body angular acceleration (Matlab)
	# =====================================================
	omega_I_omega = np.cross(omega_body, I_body @ omega_body)
	domega_body = np.linalg.solve(I_body, T_total_body - omega_I_omega)
	domega_body = np.clip(domega_body, -80.0, 80.0)

	# dead-zone for yaw to avoid long-term drift
	if abs(rpy_body[2]) < 1e-5 and abs(omega_body[2]) < 1e-5:
		domega_body[2] = 0.0

	return ddq, ddp_body, domega_body, foot_z, contact_Fz

