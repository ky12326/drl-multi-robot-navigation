import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from rclpy.qos import QoSDurabilityPolicy, QoSHistoryPolicy, QoSReliabilityPolicy
from rclpy.qos import QoSProfile
from nav_msgs.msg import Odometry
from std_srvs.srv import Empty
from gazebo_msgs.srv import SetEntityState
from gazebo_msgs.msg import ModelStates
from geometry_msgs.msg import Pose, Twist
from visualization_msgs.msg import Marker
from rclpy.logging import LoggingSeverity

SEVERITY = LoggingSeverity.ERROR


class SensorSubscriber(Node):
    def __init__(self, namespace=None):
        node_name = f"sensor_subscriber_{namespace}" if namespace else "sensor_subscriber"
        super().__init__(node_name)
        self.get_logger().set_level(SEVERITY)
        
        # 使用命名空间创建话题（添加前导斜杠）
        scan_topic = f"/{namespace}/scan" if namespace else "/scan"
        odom_topic = f"/{namespace}/odom" if namespace else "/odom"
        
        self.subscriber_scan = self.create_subscription(
            LaserScan, scan_topic, self.scan_listener_callback, 1
        )
        self.subscriber_odom = self.create_subscription(
            Odometry, odom_topic, self.odom_listener_callback, 1
        )
        self.latest_position = None
        self.latest_heading = None
        self.latest_scan = None
        self.scan_updated = False
        self.odom_updated = False

    def scan_listener_callback(self, msg):
        self.latest_scan = msg.ranges[:]
        self.scan_updated = True

    def odom_listener_callback(self, msg):
        self.latest_position = msg.pose.pose.position
        self.latest_heading = msg.pose.pose.orientation
        self.odom_updated = True

    def get_latest_sensor(self):
        return self.latest_scan, self.latest_position, self.latest_heading

    def consume_latest_sensor(self):
        """
        返回当前缓存，同时告诉调用方这次是否真的收到了新消息。
        调用后会把 updated 标记清零。
        """
        result = (
            self.latest_scan,
            self.latest_position,
            self.latest_heading,
            self.scan_updated,
            self.odom_updated,
        )
        self.scan_updated = False
        self.odom_updated = False
        return result
        
class ScanSubscriber(Node):
    def __init__(self, namespace=None):
        node_name = f"scan_subscriber_{namespace}" if namespace else "scan_subscriber"
        super().__init__(node_name)
        self.get_logger().set_level(SEVERITY)
        
        scan_topic = f"/{namespace}/scan" if namespace else "/scan"
        self.subscriber_ = self.create_subscription(
            LaserScan, scan_topic, self.listener_callback, 1
        )
        self.latest_scan = None

    def listener_callback(self, msg):
        self.latest_scan = msg.ranges[:]

    def get_latest_scan(self):
        return self.latest_scan


class OdomSubscriber(Node):
    def __init__(self, namespace=None):
        node_name = f"odom_subscriber_{namespace}" if namespace else "odom_subscriber"
        super().__init__(node_name)
        self.get_logger().set_level(SEVERITY)
        
        odom_topic = f"/{namespace}/odom" if namespace else "/odom"
        self.subscriber_ = self.create_subscription(
            Odometry, odom_topic, self.listener_callback, 1
        )
        self.latest_position = None
        self.latest_heading = None

    def listener_callback(self, msg):
        self.latest_position = msg.pose.pose.position
        self.latest_heading = msg.pose.pose.orientation

    def get_latest_odom(self):
        return self.latest_position, self.latest_heading

class GazeboModelStateSubscriber(Node):
    def __init__(self):
        super().__init__("gazebo_model_state_subscriber")
        self.get_logger().set_level(SEVERITY)

        # BEST_EFFORT required — libgazebo_ros_state.so publishes with
        # BEST_EFFORT, and RELIABLE (ROS2 default) subscribers won't match.
        _qos = QoSProfile(depth=10, reliability=QoSReliabilityPolicy.BEST_EFFORT)
        self.subscriber_ = self.create_subscription(
            ModelStates,
            "/gazebo/model_states",
            self.listener_callback,
            _qos,
        )

        self.latest_names = []
        self.latest_poses = []
        self.latest_twists = []
        self.model_updated = False

    def listener_callback(self, msg):
        self.latest_names = list(msg.name)
        self.latest_poses = list(msg.pose)
        self.latest_twists = list(msg.twist)
        self.model_updated = True

    def get_model_pose(self, model_name):
        if not self.latest_names:
            return None, None

        try:
            idx = self.latest_names.index(model_name)
            pose = self.latest_poses[idx]
            return pose.position, pose.orientation
        except ValueError:
            return None, None

    def consume_model_updated(self):
        """返回这次是否收到了新的 /gazebo/model_states，并清零标记。"""
        updated = self.model_updated
        self.model_updated = False
        return updated
        
class ResetWorldClient(Node):
    def __init__(self, namespace=None):
        node_name = f"reset_world_client_{namespace}" if namespace else "reset_world_client"
        super().__init__(node_name)
        self.get_logger().set_level(SEVERITY)
        self.reset_client = self.create_client(Empty, "/reset_world")

        self.wait_for_service(self.reset_client, "reset_world")

    def wait_for_service(self, client, service_name, timeout=10.0):
        self.get_logger().info(f"Waiting for {service_name} service...")
        if not client.wait_for_service(timeout_sec=timeout):
            self.get_logger().error(
                f"Service {service_name} not available after waiting."
            )
            raise RuntimeError(f"Service {service_name} not available.")

    def reset_world(self):
        self.get_logger().info("Calling /gazebo/reset_world service...")
        request = Empty.Request()
        future = self.reset_client.call_async(request)
        rclpy.spin_until_future_complete(self, future)
        if future.result() is not None:
            self.get_logger().info("World reset successfully.")
        else:
            self.get_logger().error(f"Failed to reset world: {future.exception()}")


class PhysicsClient(Node):
    def __init__(self, namespace=None):
        node_name = f"physics_client_{namespace}" if namespace else "physics_client"
        super().__init__(node_name)
        self.get_logger().set_level(SEVERITY)
        self.unpause_client = self.create_client(Empty, "/unpause_physics")
        self.pause_client = self.create_client(Empty, "/pause_physics")

        self.wait_for_service(self.unpause_client, "unpause_physics")
        self.wait_for_service(self.pause_client, "pause_physics")

    def wait_for_service(self, client, service_name, timeout=10.0):
        self.get_logger().info(f"Waiting for {service_name} service...")
        if not client.wait_for_service(timeout_sec=timeout):
            self.get_logger().error(
                f"Service {service_name} not available after waiting."
            )
            raise RuntimeError(f"Service {service_name} not available.")

    def pause_physics(self):
        self.get_logger().info("Calling /gazebo/pause_physics service...")
        request = Empty.Request()
        future = self.pause_client.call_async(request)
        rclpy.spin_until_future_complete(self, future)
        if future.result() is not None:
            self.get_logger().info("Physics paused successfully.")
        else:
            self.get_logger().error(f"Failed to pause physics: {future.exception()}")

    def unpause_physics(self):
        self.get_logger().info("Calling /gazebo/unpause_physics service...")
        request = Empty.Request()
        future = self.unpause_client.call_async(request)

        rclpy.spin_until_future_complete(self, future)
        if future.result() is not None:
            self.get_logger().info("Physics unpaused successfully.")
        else:
            self.get_logger().error(f"Failed to unpause physics: {future.exception()}")


class SetModelStateClient(Node):
    def __init__(self, namespace=None):
        node_name = f"set_entity_state_client_{namespace}" if namespace else "set_entity_state_client"
        super().__init__(node_name)
        self.get_logger().set_level(SEVERITY)
        self.client = self.create_client(SetEntityState, "/gazebo/set_entity_state")
        while not self.client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info("Service not available, waiting again...")
        self.request = SetEntityState.Request()

    def set_state(self, name, new_pose):
        # 设置目标实体名与位姿
        self.request.state.name = name
        self.request.state.pose = new_pose
        # 将速度清零，避免瞬移产生惯性导致的抖动/倾斜
        self.request.state.twist.linear.x = 0.0
        self.request.state.twist.linear.y = 0.0
        self.request.state.twist.linear.z = 0.0
        self.request.state.twist.angular.x = 0.0
        self.request.state.twist.angular.y = 0.0
        self.request.state.twist.angular.z = 0.0
        # 以世界坐标作为参考，防止串扰
        self.request.state.reference_frame = "world"
        # 同步等待服务完成，确保状态已被设置
        future = self.client.call_async(self.request)
        rclpy.spin_until_future_complete(self, future, timeout_sec=1.0)


class CmdVelPublisher(Node):
    def __init__(self, namespace=None):
        node_name = f"cmd_vel_publisher_{namespace}" if namespace else "cmd_vel_publisher"
        super().__init__(node_name)
        self.get_logger().set_level(SEVERITY)

        cmd_vel_topic = f"/{namespace}/cmd_vel" if namespace else "/cmd_vel"
        self.publisher_ = self.create_publisher(Twist, cmd_vel_topic, 1)

    def publish_cmd_vel(self, linear_velocity=0.0, angular_velocity=0.0):
        twist_msg = Twist()
        twist_msg.linear.x = float(linear_velocity)
        twist_msg.angular.z = float(angular_velocity)
        self.publisher_.publish(twist_msg)


class MarkerPublisher(Node):
    def __init__(self, namespace=None):
        node_name = f"marker_publisher_{namespace}" if namespace else "marker_publisher"
        super().__init__(node_name)
        self.get_logger().set_level(SEVERITY)
        
        # 修复：从 'robot1', 'robot2' 等命名空间中正确提取ID
        if namespace and 'robot' in namespace:
            try:
                # 移除 'robot' 前缀并转换为整数，ID从1开始，所以减1得到索引0, 1, ...
                self.robot_id = int(namespace.replace('robot', '')) - 1
            except (ValueError, IndexError):
                self.robot_id = 0
        else:
            self.robot_id = 0
        
        marker_topic = "visualization_marker"  # 全局话题以便在RViz中观察
        self.publisher = self.create_publisher(Marker, marker_topic, 1)

    def publish(self, x, y):
        marker = Marker()
        marker.header.frame_id = "odom"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = f"robot_{self.robot_id}_goal"
        marker.id = self.robot_id  # 使用机器人ID作为marker ID
        marker.type = Marker.CYLINDER
        marker.action = Marker.ADD

        marker.pose.position.x = x
        marker.pose.position.y = y
        marker.pose.position.z = 0.0
        marker.pose.orientation.x = 0.0
        marker.pose.orientation.y = 0.0
        marker.pose.orientation.z = 0.0
        marker.pose.orientation.w = 1.0

        marker.scale.x = 0.2
        marker.scale.y = 0.2
        marker.scale.z = 0.1

        # 根据机器人ID设置不同颜色
        color_map = {
            0: (1.0, 0.0, 0.0),  # 红色 - 机器人0
            1: (0.0, 0.0, 1.0),  # 蓝色 - 机器人1
            2: (1.0, 1.0, 0.0),  # 黄色 - 机器人2
            3: (1.0, 0.0, 1.0),  # 洋红色 - 机器人3
            4: (0.0, 1.0, 1.0),  # 青色 - 机器人4
            5: (0.5, 1.0, 0.0),  # 黄绿色 - 机器人5
            6: (1.0, 0.5, 0.0),  # 橙色 - 机器人6
        }
        r, g, b = color_map.get(self.robot_id, (1.0, 1.0, 1.0))  # 默认白色
        
        marker.color.a = 1.0
        marker.color.r = r
        marker.color.g = g
        marker.color.b = b

        self.publisher.publish(marker)
        self.get_logger().info(f"Publishing Marker for Robot {self.robot_id}")


def run_scan(args=None):
    rclpy.init()
    reading_laser = ScanSubscriber()
    reading_laser.get_logger().info("Hello friend!")
    rclpy.spin(reading_laser)

    reading_laser.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    run_scan()
