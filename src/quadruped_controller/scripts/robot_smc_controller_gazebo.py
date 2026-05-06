#!/usr/bin/env python3
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


def _read_vec_param(name: str, default: np.ndarray) -> np.ndarray:
	"""Read a ROS param that may be a scalar or list.

	If scalar: it will be broadcast to default.shape.
	If list: it must match default.size.
	"""
	v = rospy.get_param(name, None)
	if v is None:
		return np.asarray(default, dtype=float)
	arr = np.array(v, dtype=float)
	if arr.ndim == 0:
		return np.full_like(np.asarray(default, dtype=float), float(arr))
	arr = arr.reshape(-1)
	if arr.size != np.asarray(default).size:
		return np.asarray(default, dtype=float)
	return arr.reshape(np.asarray(default).shape)


def _blkdiag_leg4(mat3: np.ndarray) -> np.ndarray:
	"""Build a 12x12 block diagonal matrix with mat3 repeated 4 times."""
	mat3 = np.asarray(mat3, dtype=float).reshape(3, 3)
	out = np.zeros((12, 12), dtype=float)
	for i in range(4):
		r = slice(3 * i, 3 * (i + 1))
		out[r, r] = mat3
	return out


class QuadrupedSMCNode:
	def __init__(self) -> None:
		rospy.init_node("robot_smc_controller")

		# --- Params ---
		self.use_imu = bool(rospy.get_param("~use_imu", USE_IMU_DEFAULT))
		self.rate_hz = float(rospy.get_param("~rate", 250.0))
		self.startup_hold_s = float(rospy.get_param("~startup_hold_s", 1.0))
		self.torque_ramp_s = float(rospy.get_param("~torque_ramp_s", 1.0))
		self.use_gait_after_startup = bool(rospy.get_param("~use_gait_after_startup", True))
		# Force an initial behavior mode without relying on timing-sensitive Joy one-shots.
		# Values: rest/trot/crawl/stand/none
		self.startup_mode = str(rospy.get_param("~startup_mode", "stand")).strip().lower()
		self._startup_mode_applied = False
		# Debug (throttled) logging
		self.debug = bool(rospy.get_param("~debug", False))
		# Derivative estimation smoothing (needed for gait+IK numerical diff)
		self.dqd_lpf_alpha = float(rospy.get_param("~dqd_lpf_alpha", 0.05))  # 0..1
		# Optional extra viscous damping term (NOT in MATLAB SMC block)
		self.extra_damping = float(rospy.get_param("~extra_damping", 0.0))  # tau -= extra_damping * dq
		# Optional PD-hold mode (NOT in MATLAB SMC block)
		self.use_hold_pd = bool(rospy.get_param("~use_hold_pd", False))
		self.hold_err_thresh = float(rospy.get_param("~hold_err_thresh", 0.05))
		Kp_hold_default = np.full(12, 40.0, dtype=float)
		Kd_hold_default = np.full(12, 3.0, dtype=float)
		self.Kp_hold = np.diag(_read_vec_param("~Kp_hold", Kp_hold_default))
		self.Kd_hold = np.diag(_read_vec_param("~Kd_hold", Kd_hold_default))
		# Optional deadzone for measured dq (noise suppression)
		self.dq_deadzone = float(rospy.get_param("~dq_deadzone", 0.07))

		# --- SMC gains (match MATLAB defaults, but configurable via params) ---
		Lambda_diag_default = np.array([8.0, 12.0, 12.0], dtype=float)
		K_diag_default = np.array([1.0, 3.0, 3.0], dtype=float)
		phi_leg_default = np.array([0.15, 0.15, 0.15], dtype=float)
		Lambda_diag = _read_vec_param("~Lambda_leg_diag", Lambda_diag_default).reshape(3)
		K_diag = _read_vec_param("~K_leg_diag", K_diag_default).reshape(3)
		phi_leg = _read_vec_param("~phi_leg", phi_leg_default).reshape(3)
		self.Lambda = _blkdiag_leg4(np.diag(Lambda_diag))
		self.K = _blkdiag_leg4(np.diag(K_diag))
		self.phi = np.tile(phi_leg, 4).reshape(12)

		# --- Torque limits (match MATLAB defaults, but configurable via params) ---
		tau_max_leg_default = np.array([20.0, 55.0, 55.0], dtype=float)
		tau_max_leg = _read_vec_param("~tau_max_leg", tau_max_leg_default).reshape(3)
		self.tau_max = np.tile(tau_max_leg, 4).reshape(12)

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
		rospy.Subscriber("/quadruped_joy/joy_ramped", Joy, self.robot.joystick_command, queue_size=1)
		if self.use_imu:
			rospy.Subscriber(
				"quadruped_imu/base_link_orientation",
				Imu,
				self.robot.imu_orientation,
				queue_size=1,
			)

		self.joint_names = [
			"FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
			"FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
			"RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
			"RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
		]

		# Desired trajectory state (for numerical derivatives)
		self._qd_prev: np.ndarray | None = None
		self._dqd_prev: np.ndarray | None = None
		self._dqd_filt: np.ndarray | None = None
		self._ddqd_filt: np.ndarray | None = None
		self._t_prev: rospy.Time | None = None
		self._startup_t0: rospy.Time | None = None
		self._startup_q_hold: np.ndarray | None = None
		self._log_t_last = rospy.Time(0)

	def _apply_startup_mode(self) -> None:
		"""Apply the startup behavior mode once."""
		if self._startup_mode_applied:
			return
		mode = self.startup_mode
		if mode in ("", "none"):
			self._startup_mode_applied = True
			return

		# Clear any pending events
		self.robot.command.trot_event = False
		self.robot.command.crawl_event = False
		self.robot.command.stand_event = False
		self.robot.command.rest_event = False

		if mode == "rest":
			self.robot.command.rest_event = True
		elif mode == "trot":
			self.robot.command.trot_event = True
		elif mode == "crawl":
			self.robot.command.crawl_event = True
		elif mode == "stand":
			self.robot.command.stand_event = True
		else:
			rospy.logwarn(f"SMC startup_mode='{self.startup_mode}' not recognized; using 'stand'")
			self.robot.command.stand_event = True

		# Apply transition.
		self.robot.change_controller()
		if self.debug:
			rospy.loginfo(f"SMC startup_mode applied: {mode}, behavior_state={self.robot.state.behavior_state}")
		self._startup_mode_applied = True

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

		if msg.name:
			# luôn đảm bảo map đúng
			self._joint_name_to_index = {name: i for i, name in enumerate(msg.name)}

	def _get_q_dq(self):
		msg = self._joint_state
		if msg is None:
			rospy.logwarn("NO JOINT STATE MSG")
			return None

		if self._joint_name_to_index is None:
			rospy.logwarn("NO NAME->INDEX MAP")
			return None

		try:
			name_to_index = self._joint_name_to_index

			q = np.array([msg.position[name_to_index[j]] for j in self.joint_names])
			dq = np.array([msg.velocity[name_to_index[j]] for j in self.joint_names])

			return q, dq

		except KeyError as e:
			rospy.logwarn(f"JOINT NOT FOUND: {e}")
			return None

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

		# --- Handle time reset ---
		if self._t_prev is not None and now < self._t_prev:
			rospy.logwarn("SMC: ROS time moved backwards; resetting state")
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

		if dt <= 0.0:
			return
		if dt > 0.05:
			dt = 0.05

		# --- Get joint state ---
		state = self._get_q_dq()
		if state is None:
			return
		q, dq = state

		# --- Apply startup mode once ---
		if not self._startup_mode_applied:
			self._apply_startup_mode()

		# --- ALWAYS run state machine (CRITICAL FIX) ---
		leg_positions = self.robot.run()
		self.robot.change_controller()

		# --- Deadzone ---
		dq_dead = dq.copy()
		if self.dq_deadzone > 0.0:
			dq_dead[np.abs(dq_dead) < self.dq_deadzone] = 0.0

		# --- Startup hold ---
		if self._startup_t0 is not None:
			startup_elapsed = (now - self._startup_t0).to_sec()

			if self._startup_q_hold is None:
				self._startup_q_hold = q.copy()
				self._qd_prev = self._startup_q_hold.copy()
				self._dqd_prev = np.zeros(12)
				self._dqd_filt = np.zeros(12)
				self._ddqd_filt = np.zeros(12)

			if startup_elapsed < self.startup_hold_s:
				qd = self._startup_q_hold.copy()
				dqd = np.zeros(12)
				ddqd = np.zeros(12)
			else:
				dx, dy, dz = self.robot.state.body_local_position
				roll, pitch, yaw = self.robot.state.body_local_orientation

				try:
					qd = np.array(
						self.ik.inverse_kinematics(
							leg_positions, dx, dy, dz, roll, pitch, yaw
						),
						dtype=float,
					).reshape(12)
				except:
					rospy.logerr("IK FAILED")
					return

				dqd, ddqd = self._desired_derivatives(qd, dt)

		else:
			dx, dy, dz = self.robot.state.body_local_position
			roll, pitch, yaw = self.robot.state.body_local_orientation

			try:
				qd = np.array(
					self.ik.inverse_kinematics(
						leg_positions, dx, dy, dz, roll, pitch, yaw
					),
					dtype=float,
				).reshape(12)
			except:
				rospy.logerr("IK FAILED")
				return

			dqd, ddqd = self._desired_derivatives(qd, dt)

		# --- Optional HOLD PD ---
		if self.use_hold_pd and np.linalg.norm(qd - q) < self.hold_err_thresh:
			tau = (self.Kp_hold @ (qd - q)) - (self.Kd_hold @ dq_dead)
			tau = clip_tau(tau, self.tau_max)
			for i, pub in enumerate(self.publishers):
				pub.publish(Float64(tau[i]))
			return

		# --- SMC ---
		M, C, G = quadruped_dynamics(q, dq)

		e = q - qd
		de = dq - dqd
		s = de + (self.Lambda @ e)

		ddqr = ddqd - (self.Lambda @ de)

		sat_s = np.tanh(s / self.phi)
		tau = (M @ ddqr) + (C @ dq) + G - (self.K @ sat_s)

		# Extra damping
		if self.extra_damping > 0.0:
			tau = tau - (self.extra_damping * dq_dead)

		tau = clip_tau(tau, self.tau_max)

		# --- Torque ramp ---
		if self._startup_t0 is not None and self.torque_ramp_s > 1e-6:
			t = (now - self._startup_t0).to_sec()
			ramp = np.clip(t / self.torque_ramp_s, 0.0, 1.0)
			tau = ramp * tau

		# --- Publish ---
		for i, pub in enumerate(self.publishers):
			pub.publish(Float64(tau[i]))

		# --- Debug ---
		if self.debug:
			now2 = rospy.Time.now()
			if (now2 - self._log_t_last).to_sec() > 0.5:
				self._log_t_last = now2
				print("STATE:", self.robot.state.behavior_state)

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

