
# connect to the client
import glob # for getting latest out/000000.png
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
import time
import numpy as np
import cv2
import shutil

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
    client.set_timeout(10.0)

    world = client.get_world()

    blueprint_library = world.get_blueprint_library()

    bp = blueprint_library.filter('coupe_2020')[0]
    print(bp)

    ## ambulance set
    ambulance_bp = blueprint_library.filter('ambulance')[0]
    print(ambulance_bp)
    

    # this is for a specific spawn point: spawn_point = carla.Transform(carla.Location(x=-45,y=-130, z=40), carla.Rotation(pitch=0, yaw=180, roll=0))
    # agent vehicle spawn point [random]
    spawn_point = random.choice(world.get_map().get_spawn_points())
    vehicle = world.spawn_actor(bp, spawn_point)
    
    # ambulance spawn point [random]
    amb_spawn_point = random.choice(world.get_map().get_spawn_points())
    ambulance = world.spawn_actor(ambulance_bp, amb_spawn_point)
    ambulance.set_light_state(carla.VehicleLightState.Special1)
    ambulance.set_autopilot(True)
    # apply control to main vehicle
    vehicle.apply_control(carla.VehicleControl(throttle=1.0, steer=0.0))

    # vehicle.set_autopilot(True)  # for some NPCs to drive
    actor_list.append(ambulance)
    actor_list.append(vehicle)

    # https://carla.readthedocs.io/en/latest/cameras_and_sensors
    # get the blueprint for this sensor
    blueprint = blueprint_library.find('sensor.camera.rgb')
    # change the dimensions of the image
    blueprint.set_attribute('image_size_x', f'{IM_WIDTH}')
    blueprint.set_attribute('image_size_y', f'{IM_HEIGHT}')
    blueprint.set_attribute('fov', '110')

    # Adjust sensor relative to vehicle 
    spawn_point = carla.Transform(carla.Location(x=2.5, y=0, z=0.7))

    # spawn the sensor and attach to vehicle
    sensor = world.spawn_actor(blueprint, spawn_point, attach_to=vehicle)

    # add sensor to list of actors
    actor_list.append(sensor)



    # do something with this sensor
    #sensor.listen(lambda data: process_img(data))
    shutil.rmtree('RGB_Collect', ignore_errors=False, onerror=None) # delete out file and contents

    sensor.listen(lambda image: (image.save_to_disk('RGB_Collect/%06d.png' % image.frame)
                  if image.frame % 60 == 0 else None, process_img(image))) # RECORD PICTURES FROM CAM SENSOR
    
    print(actor_list)
    time.sleep(8)

finally:
    for actor in actor_list:
        actor.destroy()
    print('done')

# import analyze_rgb_output