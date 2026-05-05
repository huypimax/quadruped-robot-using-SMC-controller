#!/usr/bin/env python3
"""SMC joint-torque controller for the quadruped in Gazebo.

Pipeline:
  gait/trajectory (foot) -> IK -> q_des
  read joint_states -> q, dq
  SMC + (simplified) dynamics -> tau
  publish tau to effort controllers

This ports the Matlab SMC law you provided:
  e = q - qd
  de = dq - dqd
  s = de + Lambda*e
  ddqr = ddqd - Lambda*de
  tau = M*ddqr + C*dq + G - K*sat(s/phi)
  tau = clip(tau, tau_max)
"""

from __future__ import annotations

import numpy as np
import rospy

from sensor_msgs.msg import Imu, Joy, JointState
from std_msgs.msg import Float64

from RobotController import RobotController
from InverseKinematics import robot_IK
from Dynamics.quadruped_dynamics import quadruped_dynamics, tau_limit_vector, clip_tau


USE_IMU_DEFAULT = True


def _sat(x: np.ndarray) -> np.ndarray:
	"""Elementwise saturation to [-1, 1]."""
	return np.clip(x, -1.0, 1.0)


class QuadrupedSMCNode:
	def __init__(self) -> None:
		rospy.init_node("robot_smc_controller")

		# --- Params ---
		self.use_imu = bool(rospy.get_param("~use_imu", USE_IMU_DEFAULT))
		self.rate_hz = float(rospy.get_param("~rate", 250.0))
		self.startup_hold_s = float(rospy.get_param("~startup_hold_s", 1.0))
		self.torque_ramp_s = float(rospy.get_param("~torque_ramp_s", 1.0))
		self.use_gait_after_startup = bool(rospy.get_param("~use_gait_after_startup", True))
		# Chatter reduction (esp. when standing still)
		self.damping = float(rospy.get_param("~damping", 4.0))  # tau -= damping * dq
		self.dqd_lpf_alpha = float(rospy.get_param("~dqd_lpf_alpha", 0.2))  # 0..1, higher = less smoothing

		# Gains from Matlab
		Lambda_leg = np.diag([20.0, 25.0, 25.0])
		K_leg = np.diag([1.0, 3.0, 3.0])
		phi_leg = np.array([0.1, 0.1, 0.1], dtype=float)

		self.Lambda = np.zeros((12, 12), dtype=float)
		self.K = np.zeros((12, 12), dtype=float)
		for i in range(4):
			r = slice(3 * i, 3 * (i + 1))
			self.Lambda[r, r] = Lambda_leg
			self.K[r, r] = K_leg
		self.phi = np.tile(phi_leg, 4)

		self.tau_max = tau_limit_vector()

		# Robot geometry (same as robot_controller_gazebo.py)
		body = [0.366, 0.094]
		legs = [0.0, 0.08505, 0.2, 0.2]

		self.robot = RobotController.Robot(body, legs, self.use_imu)
		self.ik = robot_IK.InverseKinematics(body, legs)

		# --- ROS I/O ---
		self._joint_state: JointState | None = None
		self._joint_name_to_index: dict[str, int] | None = None

		# use controller namespace consistent with simulation.launch
		self.command_topics = [
			"/quadruped_gazebo/FR_hip_joint/command",
			"/quadruped_gazebo/FR_thigh_joint/command",
			"/quadruped_gazebo/FR_calf_joint/command",
			"/quadruped_gazebo/FL_hip_joint/command",
			"/quadruped_gazebo/FL_thigh_joint/command",
			"/quadruped_gazebo/FL_calf_joint/command",
			"/quadruped_gazebo/RR_hip_joint/command",
			"/quadruped_gazebo/RR_thigh_joint/command",
			"/quadruped_gazebo/RR_calf_joint/command",
			"/quadruped_gazebo/RL_hip_joint/command",
			"/quadruped_gazebo/RL_thigh_joint/command",
			"/quadruped_gazebo/RL_calf_joint/command",
		]
		self.publishers = [rospy.Publisher(t, Float64, queue_size=0) for t in self.command_topics]

		# Subscribers
		rospy.Subscriber("/quadruped_gazebo/joint_states", JointState, self._on_joint_state, queue_size=1)
		rospy.Subscriber("quadruped_joy/joy_ramped", Joy, self.robot.joystick_command, queue_size=1)
		if self.use_imu:
			rospy.Subscriber(
				"quadruped_imu/base_link_orientation",
				Imu,
				self.robot.imu_orientation,
				queue_size=1,
			)

		# Desired trajectory state (for numerical derivatives)
		self._qd_prev: np.ndarray | None = None
		self._dqd_prev: np.ndarray | None = None
		self._dqd_filt: np.ndarray | None = None
		self._ddqd_filt: np.ndarray | None = None
		self._t_prev: rospy.Time | None = None
		self._startup_t0: rospy.Time | None = None
		self._startup_q_hold: np.ndarray | None = None

		# Joint ordering expected by controllers/topics
		self.expected_joint_order = [
			"FR_hip_joint",
			"FR_thigh_joint",
			"FR_calf_joint",
			"FL_hip_joint",
			"FL_thigh_joint",
			"FL_calf_joint",
			"RR_hip_joint",
			"RR_thigh_joint",
			"RR_calf_joint",
			"RL_hip_joint",
			"RL_thigh_joint",
			"RL_calf_joint",
		]

	def _on_joint_state(self, msg: JointState) -> None:
		self._joint_state = msg
		if self._joint_name_to_index is None and msg.name:
			self._joint_name_to_index = {name: i for i, name in enumerate(msg.name)}

	def _get_q_dq(self) -> tuple[np.ndarray, np.ndarray] | None:
		"""Return (q, dq) in expected joint order."""
		if self._joint_state is None or self._joint_name_to_index is None:
			return None

		msg = self._joint_state
		idx = self._joint_name_to_index
		try:
			q = np.array([msg.position[idx[n]] for n in self.expected_joint_order], dtype=float)
			dq = np.array([msg.velocity[idx[n]] for n in self.expected_joint_order], dtype=float)
		except Exception:
			return None

		if q.shape != (12,) or dq.shape != (12,):
			return None
		return q, dq

	def _compute_desired_q(self) -> np.ndarray | None:
		"""Compute desired joint angles qd from gait+IK."""
		# If the user wants to keep the robot "standing" initially,
		# we can freeze the desired joint angles to the current posture.
		# This prevents a large initial jump in desired angles (common cause of falling)
		# while the simulator/controller is still settling.
		if not self.use_gait_after_startup and self._startup_q_hold is not None:
			return self._startup_q_hold.copy()

		leg_positions = self.robot.run()
		self.robot.change_controller()

		dx, dy, dz = self.robot.state.body_local_position
		roll, pitch, yaw = self.robot.state.body_local_orientation

		try:
			qd = np.array(
				self.ik.inverse_kinematics(leg_positions, dx, dy, dz, roll, pitch, yaw),
				dtype=float,
			).reshape(12)
		except Exception:
			return None
		return qd

	def _desired_derivatives(self, qd: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray]:
		"""Numerically estimate (dqd, ddqd)."""
		if self._qd_prev is None or dt <= 0.0:
			dqd = np.zeros(12)
			ddqd = np.zeros(12)
		else:
			dqd = (qd - self._qd_prev) / dt
			if self._dqd_prev is None:
				ddqd = np.zeros(12)
			else:
				ddqd = (dqd - self._dqd_prev) / dt

		# Low-pass filter to reduce numerical jitter (helps torque chatter when standing)
		a = float(np.clip(self.dqd_lpf_alpha, 0.0, 1.0))
		if self._dqd_filt is None:
			self._dqd_filt = dqd.copy()
			self._ddqd_filt = ddqd.copy()
		else:
			self._dqd_filt = a * dqd + (1.0 - a) * self._dqd_filt
			self._ddqd_filt = a * ddqd + (1.0 - a) * self._ddqd_filt

		self._qd_prev = qd.copy()
		self._dqd_prev = dqd.copy()
		return self._dqd_filt.copy(), self._ddqd_filt.copy()

	def step(self) -> None:
		now = rospy.Time.now()
		# Gazebo can reset /clock when the sim is reset/restarted.
		# If time jumped backwards, reset differentiation + startup state.
		if self._t_prev is not None and now < self._t_prev:
			rospy.logwarn("SMC: ROS time moved backwards; resetting internal timing/state")
			self._t_prev = None
			self._startup_t0 = None
			self._startup_q_hold = None
			self._qd_prev = None
			self._dqd_prev = None
			self._dqd_filt = None
			self._ddqd_filt = None
			return
		if self._t_prev is None:
			self._t_prev = now
			self._startup_t0 = now
			return
		dt = (now - self._t_prev).to_sec()
		self._t_prev = now

		# Use a sane dt if simulator stalls
		if dt <= 0.0:
			return
		if dt > 0.05:
			dt = 0.05

		state = self._get_q_dq()
		if state is None:
			return
		q, dq = state

		# --- Deadzone for velocity (remove noise when standing) ---
		dq_dead = dq.copy()
		dq_dead[np.abs(dq_dead) < 0.07] = 0.0

		# Startup: hold current posture (qd = q) for a short time.
		if self._startup_t0 is not None:
			startup_elapsed = (now - self._startup_t0).to_sec()
			if self._startup_q_hold is None:
				# On first valid joint_states, latch hold posture.
				self._startup_q_hold = q.copy()
				# Initialize desired derivative history so dqd/ddqd start at 0.
				self._qd_prev = self._startup_q_hold.copy()
				self._dqd_prev = np.zeros(12)
				self._dqd_filt = np.zeros(12)
				self._ddqd_filt = np.zeros(12)

			# During hold, don't run gait target generation.
			if startup_elapsed < self.startup_hold_s:
				qd = self._startup_q_hold.copy()
				dqd = np.zeros(12)
				ddqd = np.zeros(12)
			else:
				qd = self._compute_desired_q()
				if qd is None:
					return
				dqd, ddqd = self._desired_derivatives(qd, dt)
		else:
			qd = self._compute_desired_q()
			if qd is None:
				return
			dqd, ddqd = self._desired_derivatives(qd, dt)
			
		# --- HOLD MODE: use PD when no motion command ---
		if np.linalg.norm(dqd) < 0.01 and np.linalg.norm(dq_dead) < 0.05:
			Kp_hold = np.diag([40.0]*12)
			Kd_hold = np.diag([3.0]*12)
			tau = Kp_hold @ (qd - q) - Kd_hold @ dq_dead
			tau = clip_tau(tau, self.tau_max)
			for i, pub in enumerate(self.publishers):
				pub.publish(Float64(tau[i]))
			return
		# --- SMC law (ported from Matlab) ---
		M, C, G = quadruped_dynamics(q, dq_dead)

		e = q - qd
		de = dq_dead - dqd
		s = de + (self.Lambda @ e)
		# --- Deadzone for sliding surface (kill micro-chattering) ---
        # s[np.abs(s) < 0.02] = 0.0

		ddqr = ddqd - (self.Lambda @ de)

		sat_s = np.tanh(0.3 * s / self.phi)
		tau = (M @ ddqr) + (C @ dq_dead) + G - (self.K @ sat_s)

		# Add a small viscous damping term to reduce oscillations/chatter.
		# This is especially helpful during standstill when dq noise feeds into the controller.
		if self.damping > 0.0:
			tau = tau - (self.damping * dq_dead)

		tau = clip_tau(tau, self.tau_max)

		# Ramp torques in after startup to avoid impulse-like commands.
		if self._startup_t0 is not None and self.torque_ramp_s > 1e-6:
			t = (now - self._startup_t0).to_sec()
			ramp = np.clip(t / self.torque_ramp_s, 0.0, 1.0)
			tau = ramp * tau

		for i, pub in enumerate(self.publishers):
			pub.publish(Float64(tau[i]))

	def run(self) -> None:
		rate = rospy.Rate(self.rate_hz)
		rospy.loginfo(
			f"SMC controller running. use_imu={self.use_imu}, rate={self.rate_hz}Hz"
		)
		while not rospy.is_shutdown():
			try:
				self.step()
				rate.sleep()
			except rospy.exceptions.ROSTimeMovedBackwardsException:
				rospy.logwarn("SMC: ROSTimeMovedBackwardsException; resyncing time")
				self._t_prev = None
				continue


def main() -> None:
	node = QuadrupedSMCNode()
	node.run()


if __name__ == "__main__":
	main()

