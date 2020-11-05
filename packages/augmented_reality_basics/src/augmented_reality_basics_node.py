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
        #extrinsicsTODO
        #intrinsicsTODO
        
        #Load Map
        rospy.loginfo("[AugmentedRealityBasics]: Loading Map ...")
        self.mapName = "none"
        #loadmapTODO

        # Subscribers
        rospy.loginfo("[AugmentedRealityBasics]: Initializing Subscribers ...")
        self.imageSubscriber = rospy.Subscriber('camera_node/image/compressed', CompressedImage, self.callback)
        
        # Publishers
        rospy.loginfo("[AugmentedRealityBasics]: Initializing Publishers ...")
        self.modifiedImagePublisher = rospy.Publisher(f'augmented_reality_basics_node/{self.mapName}/image/compressed', CompressedImage, queue_size=10)

        rospy.loginfo("[AugmentedRealityBasics]: Initialized.")

    def callback(self, imgmsg):

        #TODO: process image

        return

    
    def draw_segment(self, image, pt_x, pt_y, color):
        
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

    def readYamlFile(self,fname):
        """
        Reads the YAML file in the path specified by 'fname'.
        E.G. :
            the calibration file is located in : `/data/config/calibrations/filename/DUCKIEBOT_NAME.yaml`
        """
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
        rate = rospy.Rate(1.0)
        while not rospy.is_shutdown():

            rate.sleep()


if __name__ == '__main__':
    node = AugmentedRealityBasics(node_name='augmented_reality_basics_node')
    # Keep it spinning to keep the node alive
    rospy.loginfo("[AugmentedRealityBasics]: Node is up and running!")
    node.run()

    rospy.spin()
    rospy.loginfo("[AugmentedRealityBasics]: node is up and running...")