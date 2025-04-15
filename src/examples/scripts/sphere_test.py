import torch
import rospy
from sensor_msgs.msg import JointState
from visualization_msgs.msg import Marker, MarkerArray

from utils.kinematics import MMKinematics
from utils.collision_detection import SphereNNModel


def init_joint_pub():
    joint_pub = rospy.Publisher("/sim_joint_states", JointState, queue_size=1)
    joint_msg = JointState()
    joint_msg.header.frame_id = "world"
    joint_msg.name = [
        "x_base_joint",
        "y_base_joint",
        "w_base_joint",
        "jaka_shoulder_pan_joint",
        "jaka_shoulder_lift_joint",
        "jaka_elbow_joint",
        "jaka_wrist_1_joint",
        "jaka_wrist_2_joint",
        "jaka_wrist_3_joint"
    ]

    return joint_pub, joint_msg


def init_sphere_pub():
    sphere_pub = rospy.Publisher("/sphere", MarkerArray, queue_size=1)

    return sphere_pub


def publish_sphere(spheres):
    global sphere_pub
    marker_array = MarkerArray()
    colors = [
        (0.0, 1.0, 0.0),  # Green
        (0.0, 0.0, 1.0),  # Blue
        (1.0, 1.0, 0.0),  # Yellow
        (1.0, 0.0, 1.0)  # Purple
    ]

    for i, sphere in enumerate(spheres):
        marker = Marker()
        marker.header.frame_id = "world"
        marker.header.stamp = rospy.Time.now()
        marker.ns = "spheres"
        marker.id = i
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose.position.x = sphere[0].item()
        marker.pose.position.y = sphere[1].item()
        marker.pose.position.z = sphere[2].item()
        marker.scale.x = sphere[3].item() * 2  # Diameter
        marker.scale.y = sphere[3].item() * 2  # Diameter
        marker.scale.z = sphere[3].item() * 2  # Diameter
        color = colors[i // 1 % len(colors)]
        marker.color.a = 1.0  # Alpha
        marker.color.r = color[0]
        marker.color.g = color[1]
        marker.color.b = color[2]
        marker_array.markers.append(marker)

    sphere_pub.publish(marker_array)


if __name__ == "__main__":
    rospy.init_node("shpere_test", anonymous=True)

    kinematics = MMKinematics()
    collision_detection = SphereNNModel(kinematics)

    q = torch.zeros(1, 9)

    sphere_pub = init_sphere_pub()
    joint_pub, joint_msg = init_joint_pub()

    rospy.sleep(1)

    rate = rospy.Rate(20)

    while not rospy.is_shutdown():
        kinematics.forward_kinematics_batch(q)
        collision_detection.forward_spheres_batch(q)

        spheres = collision_detection.link_spheres_trans.reshape(-1, 4)
        publish_sphere(spheres)

        joint_msg.header.stamp = rospy.Time.now()
        joint_msg.position = q.tolist()[0]
        joint_pub.publish(joint_msg)

        rate.sleep()
