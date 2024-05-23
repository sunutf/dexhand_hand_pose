import rclpy
from rclpy.node import Node
import cv2
import numpy as np
import mediapipe as mp
from std_msgs.msg import String

NUM_DOFS = 17
class HandPoseNode(Node):
    def __init__(self):
        super().__init__('hand_pose')
        self.publisher_ = self.create_publisher(String, 'dexhand_dof_stream', 10)
        self.timer = self.create_timer(0.1, self.timer_callback)  # 10 Hz
        self.cap = cv2.VideoCapture(0)
        self.mp_hands = mp.solutions.hands.Hands(
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5)
        
    def timer_callback(self):
        ret, frame = self.cap.read()
        if not ret:
            return
        
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.mp_hands.process(image)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                joint_angles = self.analyze_hand_landmarks(hand_landmarks)  # Directly pass the HandLandmark object
                self.publish_joint_angles(joint_angles)

    def timer_callback(self):
        ret, frame = self.cap.read()
        if not ret:
            return
        
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.mp_hands.process(image)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                joint_angles = self.analyze_hand_landmarks(hand_landmarks)
                self.publish_joint_angles(joint_angles)

    def analyze_hand_landmarks(self, landmarks):
        # Assuming landmarks is a list of MediaPipe Hand Landmark results
        if not landmarks:
            return np.zeros(NUM_DOFS)  # Return zero array if no hands detected

        # Only consider the first detected hand
        hand_landmarks = landmarks.landmark

        # Convert landmark list to an array of numpy arrays for easier calculations
        joint_xyz = np.array([[lmk.x, lmk.y, lmk.z] for lmk in hand_landmarks])

        # Create storage for the angles
        joint_angles = np.zeros(NUM_DOFS)

        # Index Finger Joints
        joint_angles[0] = 180 - self.angle_between(joint_xyz[0], joint_xyz[5], joint_xyz[6]) - 10
        joint_angles[1] = 90 - self.angle_between(joint_xyz[5], joint_xyz[6], joint_xyz[7])
        joint_angles[2] = 180 - self.angle_between(joint_xyz[6], joint_xyz[7], joint_xyz[8])

        # Middle Finger Joints
        joint_angles[3] = 180 - self.angle_between(joint_xyz[0], joint_xyz[9], joint_xyz[10])
        joint_angles[4] = self.angle_between(joint_xyz[5], joint_xyz[9], joint_xyz[10]) - 90 - 15
        joint_angles[5] = 180 - self.angle_between(joint_xyz[9], joint_xyz[10], joint_xyz[11])

        # Ring Finger Joints
        joint_angles[6] = 180 - self.angle_between(joint_xyz[0], joint_xyz[13], joint_xyz[14])
        joint_angles[7] = self.angle_between(joint_xyz[9], joint_xyz[13], joint_xyz[14]) - 90
        joint_angles[8] = 180 - self.angle_between(joint_xyz[13], joint_xyz[14], joint_xyz[15])

        # Pinky Finger Joints
        joint_angles[9] = 180 - self.angle_between(joint_xyz[0], joint_xyz[17], joint_xyz[18])
        joint_angles[10] = self.angle_between(joint_xyz[13], joint_xyz[17], joint_xyz[18]) - 90
        joint_angles[11] = 180 - self.angle_between(joint_xyz[17], joint_xyz[18], joint_xyz[19])

        # Thumb Joints
        joint_angles[12] = 180 - self.angle_between(joint_xyz[1], joint_xyz[2], joint_xyz[4])
        joint_angles[13] = 60 - self.angle_between(joint_xyz[2], joint_xyz[1], joint_xyz[5])
        joint_angles[14] = 180 - self.angle_between(joint_xyz[2], joint_xyz[3], joint_xyz[4])

        # Wrist Joints (Assuming wrist rotations are controlled and calculated)
        wrist_enabled = True
        if wrist_enabled:
            joint_angles[15] = 90 - self.angle_between(joint_xyz[13], joint_xyz[0], joint_xyz[0]+np.array([0,0,1]), plane=np.array([0,1,1]))
            joint_angles[16] = 90 - self.angle_between(joint_xyz[13], joint_xyz[0], joint_xyz[0]+np.array([1,0,0]), plane=np.array([1,1,0]))
        else:
            joint_angles[15] = 0
            joint_angles[16] = 0

        return joint_angles

    def angle_between(self, p1, p2, p3, plane=np.array([1, 1, 1])):
        ba = (p1 - p2) * plane
        bc = (p3 - p2) * plane
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        angle = np.arccos(cosine_angle)
        return np.degrees(angle)


    def publish_joint_angles(self, joint_angles):
        # Encode and send the joint angles as shown in the provided Python example
        scaled_angles = np.clip(joint_angles, -180, 180)
        encoded = np.interp(scaled_angles, [-180, 180], [0, 255]).astype(np.uint8)
        checksum = sum(encoded) % 256
        message = bytearray(encoded.tolist() + [checksum])
        
        # Convert to hex string for easy transport over ROS topics
        hex_string = message.hex()
        msg = String()
        msg.data = hex_string
        self.publisher_.publish(msg)
        self.get_logger().info(f'Published joint angles : {hex_string}')

def main(args=None):
    rclpy.init(args=args)
    node = HandPoseNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
