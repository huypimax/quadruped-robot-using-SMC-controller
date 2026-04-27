#!/usr/bin/env python3
#Author: lnotspotl

import rospy

from sensor_msgs.msg import Joy,Imu
from RobotController import RobotController
from InverseKinematics import robot_IK
from std_msgs.msg import Float64

USE_IMU = True
RATE = 60

rospy.init_node("Robot_Controller")

# Robot geometry
body = [0.366, 0.094]
legs = [0.,0.08505, 0.2, 0.2] 

quadruped_robot = RobotController.Robot(body, legs, USE_IMU)
inverseKinematics = robot_IK.InverseKinematics(body, legs)

command_topics = ["/quadruped_gazebo/FR_hip_joint/command",
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
                  "/quadruped_gazebo/RL_calf_joint/command"]

publishers = []
for i in range(len(command_topics)):
    publishers.append(rospy.Publisher(command_topics[i], Float64, queue_size = 0))

if USE_IMU:
    rospy.Subscriber("quadruped_imu/base_link_orientation",Imu,quadruped_robot.imu_orientation)
rospy.Subscriber("quadruped_joy/joy_ramped",Joy,quadruped_robot.joystick_command)

rate = rospy.Rate(RATE)

del body
del legs
del command_topics
del USE_IMU
del RATE

while not rospy.is_shutdown():
    leg_positions = quadruped_robot.run()
    quadruped_robot.change_controller()

    dx = quadruped_robot.state.body_local_position[0]
    dy = quadruped_robot.state.body_local_position[1]
    dz = quadruped_robot.state.body_local_position[2]
    
    roll = quadruped_robot.state.body_local_orientation[0]
    pitch = quadruped_robot.state.body_local_orientation[1]
    yaw = quadruped_robot.state.body_local_orientation[2]

    try:
        joint_angles = inverseKinematics.inverse_kinematics(leg_positions,
                               dx, dy, dz, roll, pitch, yaw)

        for i in range(len(joint_angles)):
            publishers[i].publish(joint_angles[i])
    except:
        pass

    rate.sleep()
