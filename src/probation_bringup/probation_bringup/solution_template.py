#!/usr/bin/env python3
import rclpy
from rclpy.node import Node

# Import necessary message types
from vision_msgs.msg import BoundingBoxArray
from geometry_msgs.msg import Twist

class GateNavigator(Node):

    def __init__(self):
        super().__init__('gate_navigator')

        # Define the subscription to the bounding box topic
        self.subscription = self.create_subscription(
            BoundingBoxArray,
            '/main_camera/detection/bounding_boxes',
            self.bounding_boxes_callback,
            10)
        
        # Define the publisher for velocity commands
        self.publisher_ = self.create_publisher(Twist, '/mavros/setpoint_velocity/cmd_vel_unstamped', 10)
        timer_period = 0.5  # seconds
        self.timer = self.create_timer(timer_period, self.search_for_gate)
        
        # Instance variable to store the latest bounding boxes
        self.bound_x = None
        self.bound_y = None
        self.bound_w = None
        self.bound_h = None
        self.bound_conf = None
        self.bound_label_id = None
        self.bound_label_name = None
        
    def bounding_boxes_callback(self, msg):
        # Check if any bounding boxes were received
        if not msg.bounding_boxes:
            self.get_logger().info('No bounding boxes received')
            return

        # Process the received bounding boxes
        for box in msg.bounding_boxes:
            self.bound_x = box.x
            self.bound_y = box.y
            self.bound_w = box.w
            self.bound_h = box.h
            self.bound_conf = box.conf
            self.bound_label_id = box.label_id
            self.bound_label_name = box.label_name

            self.get_logger().info(
                'Bounding Box: x=%.2f, y=%.2f, w=%.2f, h=%.2f, conf=%.2f, label_id=%d, label_name=%s'
                % (box.x, box.y, box.w, box.h, box.conf, box.label_id, box.label_name)
            )
            
        
            
    def search_for_gate(self):
        # TODO: Implement gate searching logic
        # Setup twist Object
        vel_cmd = Twist()
        
        # TODO: Implement a better logic to determine if the gate is detected and navigate towards it
        while self.bound_x is None or self.bound_y is None or self.bound_w is None or self.bound_h is None or self.bound_conf is None or self.bound_label_id is None or self.bound_label_name is None:
            # If no bounding boxes are detected, rotate to search for the gate
            vel_cmd.angular.z = 5  # Rotate at a constant speed
            self.get_logger().info('Searching for gate...')
            self.publisher_.publish(vel_cmd)
            return
        
        # Publish the velocity command
        # self.publisher_.publish(vel_cmd)
        # self.get_logger().info('Publishing: "%s"' % vel_cmd.linear.x)


def main(args=None):
    rclpy.init(args=args)
    node = GateNavigator()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()



if __name__ == '__main__':
    main()
