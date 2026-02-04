#!/root/miniconda3/envs/articulation/bin/python3
"""
ROS node that subscribes to current and reference pose topics from SAM2+ICG tracker
and prints them when messages are received.
"""

import rospy
from geometry_msgs.msg import Pose
from std_msgs.msg import Float32, String
from scipy.spatial.transform import Rotation as R
import numpy as np


class PoseMonitorNode:
    def __init__(self):
        rospy.init_node('monitor_node', anonymous=True)
        
        # Subscribers
        self.current_pose_sub = rospy.Subscriber(
            '/sam2_icg_tracker/current_pose', 
            Pose, 
            self.current_pose_callback
        )
        
        self.reference_pose_sub = rospy.Subscriber(
            '/sam2_icg_tracker/reference_pose', 
            Pose, 
            self.reference_pose_callback
        )
        
        self.euclidean_dist_sub = rospy.Subscriber(
            '/sam2_icg_tracker/euclidean_dist',
            Float32,
            self.euclidean_dist_callback
        )
        
        self.rotation_sub = rospy.Subscriber(
            '/sam2_icg_tracker/rotation',
            Float32,
            self.rotation_callback
        )
        
        self.debug_sub = rospy.Subscriber(
            '/sam2_icg_tracker/debug',
            String,
            self.debug_callback
        )
        
        rospy.loginfo("Monitor node initialized")
        rospy.loginfo("Subscribed to:")
        rospy.loginfo("  - /sam2_icg_tracker/current_pose")
        rospy.loginfo("  - /sam2_icg_tracker/reference_pose")
        rospy.loginfo("  - /sam2_icg_tracker/euclidean_dist")
        rospy.loginfo("  - /sam2_icg_tracker/rotation")
        rospy.loginfo("  - /sam2_icg_tracker/debug")
    
    def pose_to_string(self, pose_msg, pose_name="Pose"):
        """Convert pose message to a readable string."""
        # Extract translation
        x = pose_msg.position.x
        y = pose_msg.position.y
        z = pose_msg.position.z
        
        # Extract rotation (quaternion)
        qx = pose_msg.orientation.x
        qy = pose_msg.orientation.y
        qz = pose_msg.orientation.z
        qw = pose_msg.orientation.w
        
        # Convert quaternion to Euler angles for readability
        r = R.from_quat([qx, qy, qz, qw])
        euler = r.as_euler('xyz', degrees=True)
        
        # Format string
        pose_str = f"\n{pose_name}:\n"
        pose_str += f"  Translation: [{x:.4f}, {y:.4f}, {z:.4f}]\n"
        pose_str += f"  Rotation (quaternion): [{qx:.4f}, {qy:.4f}, {qz:.4f}, {qw:.4f}]\n"
        pose_str += f"  Rotation (Euler XYZ, deg): [{euler[0]:.2f}, {euler[1]:.2f}, {euler[2]:.2f}]\n"
        
        return pose_str
    
    def current_pose_callback(self, pose_msg):
        """Callback for current pose updates."""
        pose_str = self.pose_to_string(pose_msg, "Current Pose")
        rospy.loginfo(pose_str)
    
    def reference_pose_callback(self, pose_msg):
        """Callback for reference pose updates."""
        pose_str = self.pose_to_string(pose_msg, "Reference Pose")
        rospy.loginfo(pose_str)
    
    def euclidean_dist_callback(self, msg):
        """Callback for euclidean distance updates."""
        rospy.loginfo(f"\nEuclidean Distance: {msg.data:.4f} m")
    
    def rotation_callback(self, msg):
        """Callback for rotation updates."""
        rospy.loginfo(f"\nRotation: {msg.data:.2f} degrees")
        
    def debug_callback(self, msg):
        """Callback for debug updates."""
        rospy.loginfo(f"\nDebug: {msg.data}")
    
    def run(self):
        """Run the node."""
        rospy.spin()


def main():
    try:
        node = PoseMonitorNode()
        node.run()
    except rospy.ROSInterruptException:
        pass
    except Exception as e:
        rospy.logerr(f"Error in monitor node: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
