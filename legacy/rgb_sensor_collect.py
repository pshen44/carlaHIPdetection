
# connect to the client
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

# -- imports -------------------------------------------------------------------
import carla
import random
import random
import time
import numpy as np
import cv2
import shutil
# import base64
# from openai import OpenAI

SHOW_PREVIEW = False
IM_WIDTH = 640
IM_HEIGHT = 480



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
    # this is for a specific spawn point: spawn_point = carla.Transform(carla.Location(x=-45,y=-130, z=40), carla.Rotation(pitch=0, yaw=180, roll=0))
    spawn_point = random.choice(world.get_map().get_spawn_points())
    vehicle = world.spawn_actor(bp, spawn_point)
    vehicle.apply_control(carla.VehicleControl(throttle=1.0, steer=0.0))
    # vehicle.set_autopilot(True)  # for some NPCs to drive

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

    # spawn the sensor and attach to vehicle
    sensor = world.spawn_actor(blueprint, spawn_point, attach_to=vehicle)

    # add sensor to list of actors
    actor_list.append(sensor)

    # do something with this sensor
    #sensor.listen(lambda data: process_img(data))
    shutil.rmtree('out', ignore_errors=False, onerror=None) # delete out file and contents
    sensor.listen(lambda image: image.save_to_disk('out/%06d.png' % image.frame)
                  if image.frame % 60 == 0 else None) # RECORD PICTURES FROM CAM SENSOR
    
    time.sleep(8)

finally:
    for actor in actor_list:
        actor.destroy()
    print('done')