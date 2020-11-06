#!/usr/bin/env python3

import numpy as np
import os
import rospy
import yaml
from duckietown.dtros import DTROS, NodeType, TopicType, DTParam, ParamType
#from duckietown_msgs.msg import Twist2DStamped, WheelEncoderStamped, WheelsCmdStamped
from std_msgs.msg import Header, Float32
from sensor_msgs.msg import CompressedImage
#from std_srvs.srv import Trigger, TriggerResponse
import cv2
from cv_bridge import CvBridge

class AugmentedRealityBasics(DTROS):

    def __init__(self, node_name):
        """Wheel Encoder Node
        This implements basic functionality with the wheel encoders.
        """

        # Initialize the DTROS parent class
        super(AugmentedRealityBasics, self).__init__(node_name=node_name, node_type=NodeType.PERCEPTION)

        self.veh_name = rospy.get_namespace().strip("/")
        rospy.loginfo("[AugmentedRealityBasics]: Vehicle Name = %s" %self.veh_name)

        # Load Camara Calibration
        rospy.loginfo("[AugmentedRealityBasics]: Loading Camera calibration ...")
        if(not os.path.isfile(f'/data/config/calibrations/camera_intrinsic/{self.veh_name}.yaml')):
            rospy.logwarn(f'[AugmentedRealityBasics]: Could not find {self.veh_name}.yaml. Loading default.yaml')
            self.cam_calibration = self.read_yaml_file(f'/data/config/calibrations/camera_intrinsic/default.yaml')
        else:
            self.cam_calibration = self.read_yaml_file(f'/data/config/calibrations/camera_intrinsic/{self.veh_name}.yaml')
        self.cam_height = 480
        self.cam_width = 640
        
        #Load Map
        rospy.loginfo("[AugmentedRealityBasics]: Loading Map ...")
        self.map_name = os.environ.get('MAP_FILE', 'hud.yaml').strip(".yaml")
        rospy.loginfo("[AugmentedRealityBasics]: Map Name: %s" %self.map_name)
        self.map_dict = self.read_yaml_file(os.environ.get('DT_REPO_PATH', '/') + '/packages/augmented_reality_basics/maps/' + self.map_name + '.yaml')

        # Subscribers
        rospy.loginfo("[AugmentedRealityBasics]: Initializing Subscribers ...")
        self.image_subscriber = rospy.Subscriber('camera_node/image/compressed', CompressedImage, self.callback)
        
        # Publishers
        rospy.loginfo("[AugmentedRealityBasics]: Initializing Publishers ...")
        self.mod_img_pub = rospy.Publisher(f'augmented_reality_basics_node/{self.map_name}/image/compressed', CompressedImage, queue_size=1)
        self.deb_1 = rospy.Publisher(f'augmented_reality_basics_node/{self.map_name}/image/debug/undistorted', CompressedImage, queue_size=1)

        self.cv_bridge = CvBridge()

        self.ground2pixel( self.map_dict['points'] )
        rospy.loginfo(f"Debug: map_dict['points'] = {self.map_dict['points']}")

        rospy.loginfo("[AugmentedRealityBasics]: Initialized.")

    def callback(self, imgmsg):

        #convert msg to cv2
        img = self.cv_bridge.compressed_imgmsg_to_cv2(imgmsg)

        #process image
        undistorted_image = self.process_image(img)
        #debug
        undistorted_image_msg = self.cv_bridge.cv2_to_compressed_imgmsg(undistorted_image)

        #project points to img pixels
        #done only once        

        #render modified image with segments
        modified_image = self.render_segments(undistorted_image, self.map_dict['segments'])
        
        modified_image_msg = self.cv_bridge.cv2_to_compressed_imgmsg(modified_image)
        modified_image_msg.header.stamp = rospy.Time.now()

        self.mod_img_pub.publish(modified_image_msg)
        return

    def render_segments(self, img, segments):

        for seg in segments:

            pt_1_string = seg['points'][0]
            pt_1 = self.map_dict['points'][pt_1_string][1]
            pt_2_string = seg['points'][1]
            pt_2 = self.map_dict['points'][pt_2_string][1]

            self.draw_segment(img, pt_1, pt_2, seg['color'])

        return img

    def read_yaml_file(self,fname):
        """
        Reads the YAML file in the path specified by 'fname'.
        E.G. :
            the calibration file is located in : `/data/config/calibrations/filename/DUCKIEBOT_NAME.yaml`
        """
        if(not os.path.isfile(fname)):
            rospy.logwarn("[AugmentedRealityBasics]: Could not find file in %s" %fname)
            return

        with open(fname, 'r') as in_file:
            try:
                yaml_dict = yaml.load(in_file)
                return yaml_dict
            except yaml.YAMLError as exc:
                self.log("YAML syntax error. File: %s fname. Exc: %s"
                         %(fname, exc), type='fatal')
                rospy.signal_shutdown()
                return

    
    def run(self):
        rate = rospy.Rate(2.0)
        while not rospy.is_shutdown():

            
            rate.sleep()


    def process_image(self, img):

        undistorted_img = img #placeholder

        return undistorted_img

    def ground2pixel(self, ground_points_dict):
        """
        Transforms point list from their reference frame to the image pixels frame.
        """
        for point in ground_points_dict.values(): #reference frame = image01
                point[0] = "image"
                point[1][0] *= self.cam_height
                point[1][1] *= self.cam_width

    def draw_segment(self, image, start_point, end_point, color):
        
        defined_colors = {
            'red': ['rgb', [1, 0, 0]],
            'green': ['rgb', [0, 1, 0]],
            'blue': ['rgb', [0, 0, 1]],
            'yellow': ['rgb', [1, 1, 0]],
            'magenta': ['rgb', [1, 0 , 1]],
            'cyan': ['rgb', [0, 1, 1]],
            'white': ['rgb', [1, 1, 1]],
            'black': ['rgb', [0, 0, 0]]}
        
        _color_type, [r, g, b] = defined_colors[color]
        
        cv2.line(image, (start_point[1], start_point[0]), (end_point[1], end_point[0]), (b * 255, g * 255, r * 255), 5)
        
        return image


if __name__ == '__main__':
    node = AugmentedRealityBasics(node_name='augmented_reality_basics_node')
    # Keep it spinning to keep the node alive
    rospy.loginfo("[AugmentedRealityBasics]: Node is up and running!")
    node.run()

    rospy.spin()
    rospy.loginfo("[AugmentedRealityBasics]: node is up and running...")