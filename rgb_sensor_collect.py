

# .step()

def reset(self) # reset to run another test for learning

def step(self, action):
    return obs, reward, done, extra_info

import glob
import os
import sys

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass


# ==============================================================================
# -- imports -------------------------------------------------------------------
# ==============================================================================
import carla
import random
import random
import time
import numpy as np
import cv2

SHOW_PREVIEW = False
IM_WIDTH = 640
IM_HEIGHT = 480

class CarEnv: # create class
    """This is the Car Environment that we are
    defining"""
    SHOW_CAM = SHOW_PREVIEW
    STEER_AMT = 1.0 # steer left
    im_width = IM_WIDTH
    im_height = IM_HEIGHT
    front_camera = None

    def __init__(self): # create method init
        self.client = carla.Client # assign fields to object
        self.client = carla.Client('localhost', 2000)
        self.client.set_timeout(2.0)

        self.world = client.get_world()

        self.blueprint_library = world.get_blueprint_library()
        self.blueprint_library.filter('model3')[0]
    def reset_car(self):
        """reset car"""
        self.collision_hist = [] # any collision will be added to vec
        self.actor_list =[] # track actors as well
        
        self.transform = random.choice(self.world.get_map().get.spawn_points())
        self.vehicle = self.world.spawn_actor(self.model_3, self.transform)

        self.rgb_cam = self.blueprint_library.find('sensor.camera.rgb')
        self.rgb_settings("image_size_x", f"{self.im_width}")
        self.rgb_settings("image_size_y", f"{self.im_height}")
        self.rgb_settings("field_of_view", f"120")

        transform = carla.transform(carla.Location(x=2.5, z=0.7))


def process_img(image):
    i = np.array(image.raw_data)
    i2 = i.reshape((IM_HEIGHT, IM_WIDTH, 4))
    i3 = i2[:, :, :3]
    cv2.imshow("", i3)
    cv2.waitKey(1)
    return i3/255.0

actor_list = []
try:
    client = carla.Client('localhost', 2000)
    client.set_timeout(2.0)

    world = client.get_world()

    blueprint_library = world.get_blueprint_library()

    bp = blueprint_library.filter('model3')[0]
    print(bp)

    spawn_point = random.choice(world.get_map().get_spawn_points())

    vehicle = world.spawn_actor(bp, spawn_point)
    vehicle.apply_control(carla.VehicleControl(throttle=1.0, steer=0.0))
    # vehicle.set_autopilot(True)  # if you just wanted some NPCs to drive

    actor_list.append(vehicle)

    # https://carla.readthedocs.io/en/latest/cameras_and_sensors
    # get the blueprint for this sensor
    blueprint = blueprint_library.find('sensor.camera.rgb')
    # change the dimensions of the image
    blueprint.set_attribute('image_size_x', f'{IM_WIDTH}')
    blueprint.set_attribute('image_size_y', f'{IM_HEIGHT}')
    blueprint.set_attribute('fov', '110')

    # Adjust sensor relative to vehicle 
    spawn_point = carla.Transform(carla.Location(x=2.5, y=0, z=2))

    # spawn the sensor and attach to vehicle.
    sensor = world.spawn_actor(blueprint, spawn_point, attach_to=vehicle)

    # add sensor to list of actors
    actor_list.append(sensor)

    # do something with this sensor
    #sensor.listen(lambda data: process_img(data))

    sensor.listen(lambda image: image.save_to_disk('out/%06d.png' % image.frame)
                  if image.frame % 60 == 0 else None) # RECORD PICTURES FROM CAM SENSOR
    
    time.sleep(8)

finally:
    for actor in actor_list:
        actor.destroy()
    print('done')
# Connect to the client vehicle_physics.pyand retrieve the world object
# client = carla.Client('localhost', 2000)
# #client.load_world('Town02')
# world = client.get_world()

# vehicle_blueprints = world.get_blueprint_library().filter('*vehicle*') # get vehicle blueprints


# spawn_points = world.get_map().get_spawn_points()# get spawn points
# for i in range(0,50):# spawn 50 random cars at spawnpoints
#     world.try_spawn_actor(random.choice(vehicle_blueprints), random.choice(spawn_points))

# for vehicle in world.get_actors().filter('*vehicle*'):
#     vehicle.set_autopilot(True)

# ego_vehicle = world.spawn_actor(random.choice(vehicle_blueprints), random.choice(spawn_points))

# # Create a transform to place the camera on top of the vehicle
# camera_init_trans = carla.Transform(carla.Location(z=1.5))
# # We create the camera through a blueprint that defines its properties
# camera_bp = world.get_blueprint_library().find('sensor.camera.rgb')
# # We spawn the camera and attach it to our ego vehicle
# camera = world.spawn_actor(camera_bp, camera_init_trans, attach_to=ego_vehicle)

# camera.listen(lambda image: image.save_to_disk('out/%06d.png' % image.frame)) # RECORD PICTURES FROM CAM SENSOR



# ego_bp = world.get_blueprint_library().find('vehicle.lincoln.mkz_2020')

# ego_bp.set_attribute('role_name', 'hero')

# ego_vehicle = world.spawn_actor(ego_bp, random.choice(spawn_points))