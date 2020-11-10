#!/usr/bin/env python3

import numpy as np
import os
import rospy
import yaml
from duckietown.dtros import DTROS, NodeType, TopicType, DTParam, ParamType
#from duckietown_msgs.msg import Twist2DStamped, WheelEncoderStamped, WheelsCmdStamped
from std_msgs.msg import Header, Float32
from sensor_msgs.msg import CompressedImage, CameraInfo
#from std_srvs.srv import Trigger, TriggerResponse
import cv2
from cv_bridge import CvBridge
from augmenter import Augmenter

class AugmentedRealityBasics(DTROS):

    def __init__(self, node_name):
        """Wheel Encoder Node
        This implements basic functionality with the wheel encoders.
        """

        # Initialize the DTROS parent class
        super(AugmentedRealityBasics, self).__init__(node_name=node_name, node_type=NodeType.PERCEPTION)

        self.veh_name = rospy.get_namespace().strip("/")
        rospy.loginfo("[AugmentedRealityBasics]: Vehicle Name = %s" %self.veh_name)


        # Intrinsics
        rospy.loginfo("[AugmentedRealityBasics]: Loading Camera Calibration Intrinsics ...")

        if(not os.path.isfile(f'/data/config/calibrations/camera_intrinsic/{self.veh_name}.yaml')):
            rospy.logwarn(f'[AugmentedRealityBasics]: Could not find {self.veh_name}.yaml. Loading default.yaml')
            camera_intrinsic = self.read_yaml_file(f'/data/config/calibrations/camera_intrinsic/default.yaml')
        else:
            camera_intrinsic = self.read_yaml_file(f'/data/config/calibrations/camera_intrinsic/{self.veh_name}.yaml')
        camera_info = self.camera_info_from_yaml(camera_intrinsic)
        #rospy.loginfo(f"[AugmentedRealityBasics]: camera_info = {camera_info}") #debug


        # Extrinsics
        rospy.loginfo("[AugmentedRealityBasics]: Loading Camera Calibration Extrinsics ...")

        if(not os.path.isfile(f'/data/config/calibrations/camera_extrinsic/{self.veh_name}.yaml')):
            rospy.logwarn(f'[AugmentedRealityBasics]: Could not find {self.veh_name}.yaml. Loading default.yaml')
            extrinsics = self.read_yaml_file(f'/data/config/calibrations/camera_extrinsic/default.yaml')
        else:
            extrinsics = self.read_yaml_file(f'/data/config/calibrations/camera_extrinsic/{self.veh_name}.yaml')
        
        homography = np.array(extrinsics["homography"]).reshape(3,3) #homography that maps axle coordinates to image frame coordinates
        homography = np.linalg.inv(homography)
        #rospy.loginfo(f"[AugmentedRealityBasics]: homography: {homography}") #debug
        
        
        # Augmenter class
        rospy.loginfo("[AugmentedRealityBasics]: Initializing Subscribers ...")
        self.augmenter = Augmenter(camera_info, homography, debug=False)

        
        # Load Map
        rospy.loginfo("[AugmentedRealityBasics]: Loading Map ...")
        self.map_name = os.environ.get('MAP_FILE', 'hud.yaml').strip(".yaml")
        rospy.loginfo("[AugmentedRealityBasics]: Map Name: %s" %self.map_name)
        self.map_dict = self.read_yaml_file(os.environ.get('DT_REPO_PATH', '/') + '/packages/augmented_reality_basics/maps/' + self.map_name + '.yaml')


        # Remap points in map_dict from their reference frame to image frame
        self.remap_points(self.map_dict["points"])


        # CV bridge
        self.cv_bridge = CvBridge()


        # Subscribers
        rospy.loginfo("[AugmentedRealityBasics]: Initializing Subscribers ...")
        self.image_subscriber = rospy.Subscriber('camera_node/image/compressed', CompressedImage, self.callback)
        
        
        # Publishers
        rospy.loginfo("[AugmentedRealityBasics]: Initializing Publishers ...")
        self.mod_img_pub = rospy.Publisher(f'augmented_reality_basics_node/{self.map_name}/image/compressed', CompressedImage, queue_size=1)

        
        rospy.loginfo("[AugmentedRealityBasics]: Initialized.")

    
    def callback(self, imgmsg):

        # Convert msg to cv2
        img = self.cv_bridge.compressed_imgmsg_to_cv2(imgmsg)

        # Process image
        undistorted_image = self.augmenter.process_image(img)
        
        # Project points to img pixels
        # (already done during node init, doesn't need to be redone)

        # Render modified image with segments
        modified_image = self.augmenter.render_segments(undistorted_image, self.map_dict)
        
        # Create ROS msg
        modified_image_msg = self.cv_bridge.cv2_to_compressed_imgmsg(modified_image)
        modified_image_msg.header.stamp = rospy.Time.now()

        self.mod_img_pub.publish(modified_image_msg)
        
        return


    def remap_points(self, points_dict):

        for item in points_dict.values():
            frame = item[0]
            point = item[1]

            if (frame=="axle"):
                item[0] = "image"
                item[1] = self.augmenter.ground2pixel(point)
            elif (frame == "image01"):
                item[0] = "image"
                item[1] = self.augmenter.image01_to_pixel(point)
            elif (frame == "image"):
                pass
            else:
                raise Exception(f"[AugmentedRealityBasics.remap_points]: Invalid frame: {frame}")
            
        #rospy.loginfo(f"[AugmentedRealityBasics]: Remapped points: {points_dict}")
 

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

    @staticmethod
    def camera_info_from_yaml(calib_data):
        """
        Express calibration data (intrinsics) as a CameraInfo instance.
        input: calib_data: dict, loaded from yaml file
        output: intrinsics as CameraInfo instance
        """
        cam_info = CameraInfo()
        cam_info.width = calib_data['image_width']
        cam_info.height = calib_data['image_height']
        cam_info.K = calib_data['camera_matrix']['data']
        cam_info.D = calib_data['distortion_coefficients']['data']
        cam_info.R = calib_data['rectification_matrix']['data']
        cam_info.P = calib_data['projection_matrix']['data']
        cam_info.distortion_model = calib_data['distortion_model']
        return cam_info

    
    def run(self):
        rate = rospy.Rate(2.0)
        while not rospy.is_shutdown():

            #Stuff to do

            rate.sleep()


    @staticmethod
    def extract_camera_data(data):
        k = np.array(data['camera_matrix']['data']).reshape(3, 3)
        d = np.array(data['distortion_coefficients']['data'])
        r = np.array(data['rectification_matrix']['data']).reshape(3, 3)
        p = np.array(data['projection_matrix']['data']).reshape(3, 4)
        width = data['image_width']
        height = data['image_height']
        distortion_model = data['distortion_model']
        return k, d, r, p, width, height, distortion_model


if __name__ == '__main__':
    node = AugmentedRealityBasics(node_name='augmented_reality_basics_node')
    # Keep it spinning to keep the node alive
    rospy.loginfo("[AugmentedRealityBasics]: Node is up and running!")
    #node.run()

    rospy.spin()
    rospy.loginfo("[AugmentedRealityBasics]: node is up and running...")