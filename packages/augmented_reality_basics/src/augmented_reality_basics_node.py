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
        self.mod_img_pub = rospy.Publisher(f'augmented_reality_basics_node/{self.map_name}/image/compressed', CompressedImage, queue_size=10)

        self.cv_bridge = CvBridge()

        rospy.loginfo("[AugmentedRealityBasics]: Initialized.")

    def callback(self, imgmsg):

        #convert msg to cv2
        img = self.cv_bridge.compressed_imgmsg_to_cv2(imgmsg)

        #process image
        undistorted_image = process_image(img)

        #project ground to img pixels
        ground_pixels = []
        for point in self.map_dict['points'].values():
            ground_pixels.append(point[1])

        img_pixels = ground2pixel(ground_pixels)

        #render modified image with segments
        modified_image = self.render_segments(undistorted_image, self.map_dict['segments'])
        
        modified_image_msg = self.cv_bridge.cv2_to_compressed_imgmsg(modified_image)
        modified_image_msg.header.stamp = rospy.Time.now()

        self.mod_img_pub.publish(modified_image_msg)
        return

    def render_segments(self, img, segments):

        for seg in segments:

            pt_x_string = seg['points'][0]
            pt_x = self.map_dict['points'][pt_x_string][1]
            pt_y_string = seg['points'][1]
            pt_y = self.map_dict['points'][pt_y_string][1]

            draw_segment(img, pt_x, pt_y, seg['color'])

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

############## Augmenter

def process_image(img):

    undistorted_img = img #placeholder

    return undistorted_img

def ground2pixel(ground_pixels):

    img_pixels = ground_pixels #placeholder

    return ground_pixels

def draw_segment(image, pt_x, pt_y, color):
        
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
        
    cv2.line(image, (pt_x[0], pt_y[0]), (pt_x[1], pt_y[1]), (b * 255, g * 255, r * 255), 5)
        
    return image

##############


if __name__ == '__main__':
    node = AugmentedRealityBasics(node_name='augmented_reality_basics_node')
    # Keep it spinning to keep the node alive
    rospy.loginfo("[AugmentedRealityBasics]: Node is up and running!")
    node.run()

    rospy.spin()
    rospy.loginfo("[AugmentedRealityBasics]: node is up and running...")