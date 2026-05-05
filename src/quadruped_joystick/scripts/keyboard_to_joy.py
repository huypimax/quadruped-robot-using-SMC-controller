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

            # Notes:
            # - This node publishes a one-shot Joy message per keypress.
            # - Axes/buttons indices are chosen to match the controllers in
            #   src/quadruped_controller/scripts/RobotController/.
            # - Mode switching is done by buttons[0..3] in RobotController.py.

            # MODE
            if key == '1':
                self.msg.buttons[0] = 1  # REST
            elif key == '2':
                self.msg.buttons[1] = 1  # TROT
            elif key == '3':
                self.msg.buttons[2] = 1  # CRAWL
            elif key == '4':
                self.msg.buttons[3] = 1  # STAND

            # Toggles used by RestController / TrotGaitController
            # - buttons[7]: toggle use_imu (roll/pitch compensation)
            # - buttons[6]: toggle autoRest (trot only)
            elif key == 'm':
                self.msg.buttons[7] = 1
            elif key == 'n':
                self.msg.buttons[6] = 1

            # Quick help
            elif key in ('h', '?'):
                rospy.loginfo(
                    "KeyboardToJoy mapping:\n"
                    "  Modes: 1=REST, 2=TROT, 3=CRAWL, 4=STAND\n"
                    "  Toggles: m=toggle IMU compensation (button[7]), n=toggle Trot autoRest (button[6])\n"
                    "  Trot/Crawl: w/s forward/back (axis[4]), a/d yaw left/right (axis[0]), q/e strafe left/right (axis[3])\n"
                    "  REST position: u/o body x +/- (axis[7]), j/l body y +/- (axis[6]), i/k body z +/- (axis[1])\n"
                    "  REST orientation: f/h roll +/- (axis[0]), t/g pitch +/- (axis[4]), r/y yaw +/- (axis[3])\n"
                    "  STAND: u/o body x +/- (axis[7]); FR: i/k X +/- (axis[1]), j/l Y +/- (axis[0]); FL: t/g X +/- (axis[4]), f/h Y +/- (axis[3])"
                )

            # -----------------------
            # Locomotion (TROT/CRAWL)
            # -----------------------
            # Trot uses:
            #   axis[4] -> v_x, axis[3] -> v_y, axis[0] -> yaw_rate
            # Crawl uses:
            #   axis[4] -> v_x, axis[0] -> yaw_rate
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

            # -----------------------
            # REST (body pose control)
            # -----------------------
            # RestController uses:
            #   position: axis[7]=x, axis[6]=y, axis[1]=z
            #   orientation: axis[0]=roll, axis[4]=pitch, axis[3]=yaw
            elif key == 'u':
                self.msg.axes[7] = 1.0
            elif key == 'o':
                self.msg.axes[7] = -1.0
            elif key == 'j':
                self.msg.axes[6] = 1.0
            elif key == 'l':
                self.msg.axes[6] = -1.0
            elif key == 'i':
                self.msg.axes[1] = 1.0
            elif key == 'k':
                self.msg.axes[1] = -1.0

            elif key == 'f':
                self.msg.axes[0] = 1.0
            elif key == 'h':
                self.msg.axes[0] = -1.0
            elif key == 't':
                self.msg.axes[4] = 1.0
            elif key == 'g':
                self.msg.axes[4] = -1.0
            elif key == 'r':
                self.msg.axes[3] = 1.0
            elif key == 'y':
                self.msg.axes[3] = -1.0

            # -----------------------
            # STAND (front legs reach)
            # -----------------------
            # StandController uses:
            #   axis[7]=body_x
            #   FR: axis[1]=X, axis[0]=Y
            #   FL: axis[4]=X, axis[3]=Y
            # We reuse the same keys as REST pose; practical usage is:
            # first press '4' to enter STAND, then use the keys below.
            # Body x:
            elif key == 'U':
                self.msg.axes[7] = 1.0
            elif key == 'O':
                self.msg.axes[7] = -1.0
            # FR leg (uppercase so it doesn't fight with rest pose keys):
            elif key == 'I':
                self.msg.axes[1] = 1.0
            elif key == 'K':
                self.msg.axes[1] = -1.0
            elif key == 'J':
                self.msg.axes[0] = 1.0
            elif key == 'L':
                self.msg.axes[0] = -1.0
            # FL leg:
            elif key == 'T':
                self.msg.axes[4] = 1.0
            elif key == 'G':
                self.msg.axes[4] = -1.0
            elif key == 'F':
                self.msg.axes[3] = 1.0
            elif key == 'H':
                self.msg.axes[3] = -1.0

            self.pub.publish(self.msg)
            rate.sleep()

if __name__ == "__main__":
    node = KeyboardToJoy()
    node.run()