#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import Joy
import sys, select, tty, termios

class KeyboardToJoy:
    def __init__(self):
        rospy.init_node("keyboard_to_joy")
        self.pub = rospy.Publisher("joy", Joy, queue_size=10)

        self.msg = Joy()
        self.msg.axes = [0.0]*8
        self.msg.buttons = [0]*11

        self.settings = termios.tcgetattr(sys.stdin)

    def get_key(self):
        tty.setraw(sys.stdin.fileno())
        rlist, _, _ = select.select([sys.stdin], [], [], 0.1)
        key = sys.stdin.read(1) if rlist else ''
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.settings)
        return key

    def run(self):
        rate = rospy.Rate(20)

        while not rospy.is_shutdown():
            key = self.get_key()

            # reset
            self.msg.axes = [0.0]*8
            self.msg.buttons = [0] * 11

            # MODE
            if key == '1':
                self.msg.buttons[0] = 1  # REST
            elif key == '2':
                self.msg.buttons[1] = 1  # TROT
            elif key == '3':
                self.msg.buttons[2] = 1  # CRAWL
            elif key == '4':
                self.msg.buttons[3] = 1  # STAND

            # TROT / CRAWL movement
            if key == 'w':
                self.msg.axes[4] = 1.0
            elif key == 's':
                self.msg.axes[4] = -1.0

            elif key == 'a':
                self.msg.axes[0] = 1.0
            elif key == 'd':
                self.msg.axes[0] = -1.0

            elif key == 'q':
                self.msg.axes[3] = 1.0
            elif key == 'e':
                self.msg.axes[3] = -1.0

            # REST body control
            elif key == 'i':
                self.msg.axes[1] = 1.0
            elif key == 'k':
                self.msg.axes[1] = -1.0

            elif key == 'j':
                self.msg.axes[6] = 1.0
            elif key == 'l':
                self.msg.axes[6] = -1.0

            elif key == 'u':
                self.msg.axes[7] = 1.0
            elif key == 'o':
                self.msg.axes[7] = -1.0

            # STAND fine control
            elif key == 'z':
                self.msg.axes[7] = 1.0
            elif key == 'x':
                self.msg.axes[7] = -1.0

            self.pub.publish(self.msg)
            rate.sleep()

if __name__ == "__main__":
    node = KeyboardToJoy()
    node.run()