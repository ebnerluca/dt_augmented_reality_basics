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

        # Load Camara Calibration
        # Intrinsics
        rospy.loginfo("[AugmentedRealityBasics]: Loading Camera Calibration Intrinsics ...")

        #if(not os.path.isfile(f'/data/config/calibrations/camera_intrinsic/{self.veh_name}.yaml')):
        #    rospy.logwarn(f'[AugmentedRealityBasics]: Could not find {self.veh_name}.yaml. Loading default.yaml')
        #    self.camera_calibration_intrinsics = self.read_yaml_file(f'/data/config/calibrations/camera_intrinsic/default.yaml')
        #else:
        #    self.camera_calibration_intrinsics = self.read_yaml_file(f'/data/config/calibrations/camera_intrinsic/{self.veh_name}.yaml')
        
        # self._K, self._D, self._R, self._P, self.cam_width, self.cam_height, self._distortion_model = self.extract_camera_data(self.camera_calibration_intrinsics)
        # self._K_rect, self._roi = cv2.getOptimalNewCameraMatrix(self._K, self._D, (self.cam_width, self.cam_height), 1)
        #rospy.loginfo(f"[AugmentedRealityBasics]: roi: {self._roi}")
        #rospy.loginfo(f"[AugmentedRealityBasics]: K_rect: \n{self._K_rect}")
        #rospy.loginfo(f"[AugmentedRealityBasics]: K: \n{self._K}")
        #rospy.loginfo(f"[AugmentedRealityBasics]: D: \n{self._D}")
        #rospy.loginfo(f"[AugmentedRealityBasics]: R: \n{self._R}")
        #rospy.loginfo(f"[AugmentedRealityBasics]: P: \n{self._P}")
        #rospy.loginfo(f"[AugmentedRealityBasics]: cam_width: {self.cam_width}")
        #rospy.loginfo(f"[AugmentedRealityBasics]: cam_height: {self.cam_height}")
        if(not os.path.isfile(f'/data/config/calibrations/camera_intrinsic/{self.veh_name}.yaml')):
            rospy.logwarn(f'[AugmentedRealityBasics]: Could not find {self.veh_name}.yaml. Loading default.yaml')
            camera_intrinsic = self.read_yaml_file(f'/data/config/calibrations/camera_intrinsic/default.yaml')
        else:
            camera_intrinsic = self.read_yaml_file(f'/data/config/calibrations/camera_intrinsic/{self.veh_name}.yaml')
        camera_info = self.camera_info_from_yaml(camera_intrinsic)
        rospy.loginfo(f"[AugmentedRealityBasics]: camera_info = {camera_info}")


        # Extrinsics
        rospy.loginfo("[AugmentedRealityBasics]: Loading Camera Calibration Extrinsics ...")

        if(not os.path.isfile(f'/data/config/calibrations/camera_extrinsic/{self.veh_name}.yaml')):
            rospy.logwarn(f'[AugmentedRealityBasics]: Could not find {self.veh_name}.yaml. Loading default.yaml')
            extrinsics = self.read_yaml_file(f'/data/config/calibrations/camera_extrinsic/default.yaml')
        else:
            extrinsics = self.read_yaml_file(f'/data/config/calibrations/camera_extrinsic/{self.veh_name}.yaml')
        
        #self.camera_calibration_extrinsics = np.array(extrinsics["homography"]).reshape(3,3)
        #self.camera_calibration_extrinsics_inv = np.linalg.inv(self.camera_calibration_extrinsics)
        homography = np.array(extrinsics["homography"]).reshape(3,3) #homography that maps axle coordinates to image frame coordinates
        homography = np.linalg.inv(homography)
        rospy.loginfo(f"[AugmentedRealityBasics]: homography: {homography}")
        
        
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
        #undistorted_image = self.process_image(img)
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
    

    #def render_segments(self, img, segments):

    #    for seg in segments:

    #        pt_1_string = seg['points'][0]
    #        pt_1 = self.map_dict['points'][pt_1_string][1]
    #        pt_2_string = seg['points'][1]
    #        pt_2 = self.map_dict['points'][pt_2_string][1]

    #        self.draw_segment(img, pt_1, pt_2, seg['color'])

    #    return img

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
        :param calib_data: dict, loaded from yaml file
        :return: intrinsics as CameraInfo instance
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

            
            rate.sleep()


    #def process_image(self, img):
    #    """
    #    Rectify image
    #    """
    #    undistorted_img = cv2.undistort(img, self._K, self._D, None, self._K_rect)
    #    # Optionally crop image to ROI
    #    #x, y, w, h = self._roi
    #    #undistorted_img = undistorted_img[y:y + h, x:x + w]
    #    return undistorted_img

    #def ground2pixel(self, ground_points_dict):
    #    """
    #    Transforms point list from their reference frame to the image pixels frame.
    #    """
    #    for point in ground_points_dict.values(): #reference frame = image01
    #            
    #            frame = point[0]

    #            if(frame == "image01"):
    #                point[0] = "image"
    #                point[1][0] *= self.cam_height
    #                point[1][1] *= self.cam_width

    #            elif(frame == "axle"):
    #                point[0] = "image"
    #                point_h = np.array([point[1][0], point[1][1], 1.0])
    #                pixel_h = np.dot(self.camera_calibration_extrinsics_inv, point_h) #point_h must be [x,y,1]
    #                pixel_h = pixel_h / pixel_h[2]
    #                pixel = pixel_h[0:2] 
    #                pixel = [int(i) for i in pixel] #pixel index must be integers
    #                point[1][0] = pixel[1]
    #                point[1][1] = pixel[0] #image frame points are [row, collumn]

    #@staticmethod
    #def draw_segment(image, start_point, end_point, color):
        
    #    defined_colors = {
    #        'red': ['rgb', [1, 0, 0]],
    #        'green': ['rgb', [0, 1, 0]],
    #        'blue': ['rgb', [0, 0, 1]],
    #        'yellow': ['rgb', [1, 1, 0]],
    #        'magenta': ['rgb', [1, 0 , 1]],
    #        'cyan': ['rgb', [0, 1, 1]],
    #        'white': ['rgb', [1, 1, 1]],
    #        'black': ['rgb', [0, 0, 0]]}
        
    #    _color_type, [r, g, b] = defined_colors[color]
        
    #    cv2.line(image, (start_point[1], start_point[0]), (end_point[1], end_point[0]), (b * 255, g * 255, r * 255), 5)
        
    #    return image

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
    node.run()

    rospy.spin()
    rospy.loginfo("[AugmentedRealityBasics]: node is up and running...")