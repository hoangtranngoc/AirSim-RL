from airsim import CarClient, CarControls, ImageRequest, ImageType
from configparser import ConfigParser
import numpy as np
from numpy.linalg import norm
from os.path import dirname, abspath, join

class CarAgent(CarClient):
    def __init__(self):
        # connect to the AirSim simulator
        super().__init__()
        super().confirmConnection()
        super().enableApiControl(True)

        # read configuration
        config = ConfigParser()
        config.read(join(dirname(dirname(abspath(__file__))), 'config.ini'))
        
        self.image_height = int(config['airsim_settings']['image_height'])
        self.image_width = int(config['airsim_settings']['image_width'])
        self.image_channels = int(config['airsim_settings']['image_channels'])
        self.image_size = self.image_height * self.image_width * self.image_channels

        self.action_mode = int(config['car_agent']['action_mode'])
        self.throttle = float(config['car_agent']['fixed_throttle'])
        self.steering_granularity = int(config['car_agent']['steering_granularity'])
        steering_max = float(config['car_agent']['steering_max'])
        self.steering_values = np.arange(-steering_max, steering_max, 2*steering_max/(self.steering_granularity-1)).tolist()
        self.steering_values.append(steering_max)
        
        # fetch waypoints
        waypoint_regex = config['airsim_settings']['waypoint_regex']
        self._fetchWayPoints(waypoint_regex)
   
    def restart(self):
        super().reset()
        super().enableApiControl(True)
        
    # get RGB image from the front camera    
    def observe(self):
        size = 0
        while size != self.image_size: # Sometimes simGetImages() return an unexpected resonpse. 
                                       # If so, try it again.
            response = super().simGetImages([ImageRequest(0, ImageType.Scene, False, False)])[0]
            img1d_rgb = np.frombuffer(response.image_data_uint8, dtype=np.uint8)
            size = img1d_rgb.size
            
        img3d_rgb = img1d_rgb.reshape(self.image_height, self.image_width, self.image_channels)
        return img3d_rgb
    
    def move(self, action):
        car_controls = self._interpret_action(action)
        super().setCarControls(car_controls)
                
    def _interpret_action(self, action):
        car_controls = CarControls()
        if (self.action_mode == 0): # change steering only, throttle is fixed
            car_controls.throttle = self.throttle
            car_controls.steering = self.steering_values[action]
        elif (self.action_mode == 1): # change both steering and throttle
            return NotImplemented
        elif (self.action_mode == 2):
            return NotImplemented
        else:
            return NotImplemented
                   
        return car_controls
    
    def _fetchWayPoints(self, waypoint_regex):
        wp_names = super().simListSceneObjects(waypoint_regex)
        wp_names.sort()
        print(wp_names)
        vec2r_to_numpy_array = lambda vec: np.array([vec.x_val, vec.y_val])
        
        self.waypoints = []
        for wp in wp_names: 
            pose = super().simGetObjectPose(wp)
            self.waypoints.append(vec2r_to_numpy_array(pose.position))
               
        return
        
    def simGetWayPoints(self):
        return self.waypoints
    
    def simGet2ClosestWayPoints(self):
        total_distance = lambda p, p1, p2: norm(p-p1) + norm(p-p2)
        pos = super().simGetVehiclePose().position
        car_point = np.array([pos.x_val, pos.y_val])
        
        min_dist = 9999999 
        min_i = 0
        for i in range(len(self.waypoints)-1):
            dist = total_distance(car_point, self.waypoints[i], self.waypoints[i+1])
            if dist < min_dist:
                min_dist = dist
                min_i = i
        
        return self.waypoints[min_i], self.waypoints[min_i+1]