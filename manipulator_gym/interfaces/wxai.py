import numpy as np
import math
import cv2
from scipy.spatial.transform import Rotation as rotation

# from interbotix_xs_modules.xs_robot.arm import InterbotixManipulatorXS
import trossen_arm
from trossen_arm import ArrayDouble6, InterpolationSpace
from manipulator_gym.interfaces.base_interface import ManipulatorInterface
from manipulator_gym.utils.utils import (
    rotationMatrixToEulerAngles,
    eulerAnglesToRotationMatrix,
)

import pyrealsense2 as rs

##############################################################################

ARM_IP = "192.168.1.4" 

REALSENSE_SERIAL = [838212073584, 409122274608]

class WidowXAIInterface(ManipulatorInterface):

    def __init__(self, cam_ids=[0], blocking_control=True):

        self.driver = trossen_arm.TrossenArmDriver()  
        
        self.driver.configure(model=trossen_arm.Model.wxai_v0, 
                              end_effector=trossen_arm.StandardEndEffector.wxai_v0_follower, 
                              serv_ip=ARM_IP, 
                              clear_error=True, 
                              timeout=20.0)
        
        self.driver.set_arm_modes(trossen_arm.Mode.position)
        self.driver.set_gripper_mode(trossen_arm.Mode.position)

        self.home_pose = [0, np.pi/ 3, np.pi / 6, np.pi / 5, 0, 0, 0]

        print("Using camera ids: ", cam_ids)
        self._caps = []
        self._realsense_list = []

        import time as _time
        ctx = rs.context()
        for id in cam_ids:
            if id < 0:
                serial_number = REALSENSE_SERIAL[abs(id) - 1]

                # hardware reset to recover from bad firmware/USB state
                for dev in ctx.query_devices():
                    if dev.get_info(rs.camera_info.serial_number) == str(serial_number):
                        dev.hardware_reset()
                        break
                _time.sleep(3)

                pipeline = rs.pipeline()
                config = rs.config()
                config.enable_device(str(serial_number))
                config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

                try:
                    pipeline.start(config)
                    self._realsense_list.append(pipeline)
                    print(f"[OK] RealSense started: {serial_number}")
                except Exception as e:
                    print(f"[ERROR] Failed to start RealSense {serial_number}: {e}")
            else:
                print(f"Using webcam id {id}")
                self._caps.append(cv2.VideoCapture(id))

        assert len(self._caps) <= 2, "Only 2 cameras are supported"
        self.blocking_control = blocking_control
        # start persistent image fetching thread to avoid latency issues
        img_primary_thread = self.start_img_fetch_thread()

    def fetch_primary_img(self):

        if len(self._realsense_list) > 0:
            frames = self._realsense_list[0].wait_for_frames()
            color_frame = frames.get_color_frame()

            if not color_frame:
                raise Exception("No RealSense frame")

            frame = np.asanyarray(color_frame.get_data())

        else:
            ret, frame = self._caps[0].read()

            if not ret:
                raise Exception("Can't receive frame")

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        self._primary_frame = cv2.resize(frame, (256, 256))

    def fetch_wrist_img(self) -> None:
        if len(self._realsense_list) == 0:
            self._wrist_frame = None

        elif len(self._realsense_list) == 1 and len(self._caps) == 1:
            frames = self._realsense_list[0].wait_for_frames()
            color_frame = frames.get_color_frame()

            if not color_frame:
                raise Exception("No RealSense frame")

            frame = np.asanyarray(color_frame.get_data())
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            self._wrist_frame = cv2.resize(frame, (256, 256))

        else:
            frames = self._realsense_list[1].wait_for_frames()
            color_frame = frames.get_color_frame()

            if not color_frame:
                raise Exception("No RealSense frame")

            frame = np.asanyarray(color_frame.get_data())
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            self._wrist_frame = cv2.resize(frame, (256, 256))


    @property
    def eef_pose(self) -> np.ndarray:
        return self._get_ee_pose()

    @property
    def gripper_state(self) -> float:
        # Exact ref implementation:
        # https://github.com/Interbotix/interbotix_ros_toolboxes/blob/0c739cdab1dbab03d79e752b43fa3db14d5bb15e/interbotix_xs_toolbox/interbotix_xs_modules/src/interbotix_xs_modules/gripper.py#L62
        gripper_pos = self.driver.get_gripper_position()

        return (gripper_pos + 0.04) / 0.08

    def step_action(self, action: np.ndarray) -> bool:
        """
        Override function from base class
        """
        # move the end effector, Not that dz is up and down, dy is left and right
        print("running action: ", action)
        action[:6] = np.clip(action[:6], -0.01, 0.01)
        self._move_eef_relative(
            dx=action[0],
            dy=action[1],
            dz=action[2],
            drx=action[3],
            dry=action[4],
            drz=action[5],
        )
        if action[6] > 0.5:
            print("open gripper", self.gripper_state)
            self.move_gripper(1.0)
        else:
            print("close gripper", self.gripper_state)
            self.move_gripper(0.0)
        return True

    def reset(self, reset_pose=True) -> bool:
        """Override function from base class"""
        print("Reset robot interface, reset to home pose?: ", reset_pose)
        if reset_pose:
            n = self.driver.get_num_joints()
            # zero out any residual velocity / compliance state before homing
            self.driver.set_all_modes(trossen_arm.Mode.velocity)
            self.driver.set_all_velocities([0.0] * n, 0.0, False)
            self.driver.set_all_modes(trossen_arm.Mode.position)
            self.driver.set_all_positions(self.home_pose, goal_time=2.0, blocking=False)
        return True

    def _move_eef_relative(self, dx=0, dy=0, dz=0, drx=0, dry=0, drz=0):
        # curr_rpy = rotationMatrixToEulerAngles(self._arm.T_sb[:3, :3])
        curr_pos = self.eef_pose

        goal_xyz = curr_pos[:3] + np.array([dx, dy, dz])
        goal_rpy = curr_pos[3:] + np.array([drx, dry, drz])
        goal_rotvec = rotation.from_euler('xyz', goal_rpy).as_rotvec()

        self.driver.set_cartesian_positions(
            ArrayDouble6(np.concatenate([goal_xyz, goal_rotvec])),
            interpolation_space=InterpolationSpace.joint,
            goal_time=0.8
        )

        # time.sleep(0.05)

    def move_eef(self, pose: np.ndarray, goal_time: float = 2.0) -> bool:
        """
        takes in a 6D pose moves the arm to the target pose
        """
        assert len(pose) == 6, "pose must be 6D"

        goal_xyz = pose[:3]
        goal_rpy = pose[3:]

        goal_rotvec = rotation.from_euler('xyz', goal_rpy).as_rotvec()

        self.driver.set_cartesian_positions(
            ArrayDouble6(np.concatenate([goal_xyz, goal_rotvec])),
            interpolation_space=InterpolationSpace.joint,
            goal_time=goal_time
        )

        self._move_eef_relative(0, 0, 0, 0, 0, 0)  # TODO FIx bug in compliance control
        
        return True

    def move_gripper(self, grip_state: float):
        if grip_state > 0:
            self.driver.set_gripper_position(0.04, goal_time=1.0, blocking=True)
        elif grip_state <= 0:
            self.driver.set_gripper_position(-0.04, goal_time=1.0, blocking=True)
        return True
    
    def _get_ee_pose(self):
        """This returns a np array of the x, y, z of the end effector"""
        pos = self.driver.get_cartesian_positions()
        xyz = pos[:3]
        rpy = rotation.from_rotvec(pos[3:]).as_euler('xyz', degrees=False)
        return np.concatenate([xyz, rpy])

    def __del__(self):
        self.driver.cleanup(reboot_controller=False)
        del self.driver



# arm = WidowXAIInterface(cam_ids=[0], blocking_control=True)
# current_pose = arm.eef_pose

# arm.move_gripper(1.0)  # 1.0 表示张开
# arm.move_gripper(0.0)  # 1.0 表示张开
# arm.move_gripper(1.0)  # 1.0 表示张开
# arm.move_gripper(0.0)  # 1.0 表示张开

# arm.move_gripper(0.0)  # 1.0 表示张开

# top_pose = np.array([0.25, 0.0, 0.2, 0.0, 0.0, 0.0])
# arm.move_eef(top_pose)

# arm._move_eef_relative(0.2, current_pose[1], 0, 0.0, 0.0, 0.0)
# arm.move_gripper(1.0)  # 1.0 表示张开
# arm.move_eef(np.array([0.2, 0.0, 0.15, 0.0, 1.57, 0.0]))

# # arm.move_eef(current_pose)
# arm.move_gripper(0.0)  # 1.0 表示张开
# for i in range (100):
#     arm._move_eef_relative(-0.00043246156327866186, -0.002042921614997577, -0.004398448224015101, -0.005247634638758394, 0.013956425710635989, 0.0015450867332663376)