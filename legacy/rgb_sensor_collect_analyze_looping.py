import glob
import os
import random
import time
import shutil
import base64
from openai import OpenAI
import carla
from datetime import datetime
import numpy as np
from PIL import Image
import io


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def analyze_image(image):
    if image.frame % 120 == 0:
        image.save_to_disk(f"RGB_Collect/{image.frame:06d}.png") # for reference

        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        array = np.reshape(array, (image.height, image.width, 4))[:, :, :3]
        pil_image = Image.fromarray(array).resize((640, 360))
        buffer = io.BytesIO()
        pil_image.save(buffer, format="JPEG")
        encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")

        gpt_prompt = """
        You are operating on an autonomous vehicle equipped with a forward-facing RGB camera sensor.

        Your task is to analyze a series of RGB camera images and identify the presence of High Illumination Priority (HIP) objects — any object emitting strong lights that must be prioritized by the vehicle. These include:
        - Emergency vehicles only WITH LIGHTS ON
        - Cars with hazard lights FLASHING AND ON
        - Any other FLASHING illuminated vehicles
                            
        Car response to HIPs is defined as any action that would require the car to slow down, stop, or change lanes.
        A response will only be required if the HIP is in the car's current road (any lane), or if the HIP is in oncoming traffic ONLY for emergency vehicles.

        Your response will ONLY be in a JSON format. Do not add anything else:

        '''json
        {
        "car_lane_position": <integer, representing the lane the car is currently in, not counting lanes with opposing traffic. Lane numbers increase from left to right [The shoulder is counted]>,
        "available_lane_positions": <list of integers starting from 1, representing all lanes currently visible on the road from left to right>,
        "HIPs": <list of strings, each string should name a HIP object, e.g., "Ambulance", "Hazard Lights", "Flashing Vehicle">,
        "car_response": <boolean, true if the car should slow down, stop, or change lanes due to the HIP; false otherwise>
        }                     
        '''
        """
        client = OpenAI()
        gpt_prompt = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": gpt_prompt
                        
                    },
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded}"}} 
                            ]
            }
                    ]

        try:
            completion = client.chat.completions.create(model="gpt-4.1",
                                                     messages=gpt_prompt,
                                                     stream=True,)
            for chunk in completion:
                if chunk.choices[0].delta.content is not None:
                    print(chunk.choices[0].delta.content, end="")
        except Exception as e:
            print(f"[ERROR] GPT call failed for frame {image.frame}: {e}")


actor_list = []
try:
    client = carla.Client('localhost', 2000)
    client.set_timeout(6.0)

    world = client.get_world()
    map = world.get_map()


    blueprint_library = world.get_blueprint_library()

    bp = blueprint_library.filter('coupe_2020')[0]
    ambulance_bp = blueprint_library.filter('ambulance')[0]

    spawn_point = random.choice(world.get_map().get_spawn_points())
    vehicle = world.spawn_actor(bp, spawn_point)
    vehicle.set_autopilot(True)

    tm = client.get_trafficmanager()
    tm_port = tm.get_port()
    num_ambulances = 3

    for _ in range(num_ambulances):
        amb_spawn_point = random.choice(world.get_map().get_spawn_points())
        ambulance = world.spawn_actor(ambulance_bp, amb_spawn_point)
        ambulance.set_light_state(carla.VehicleLightState.Special1)
        ambulance.set_autopilot(True, tm_port)
        tm.ignore_lights_percentage(ambulance, 100.0)
        tm.distance_to_leading_vehicle(ambulance,0)
        tm.vehicle_percentage_speed_difference(ambulance, -80.0)
        actor_list.append(ambulance)

    blueprint = blueprint_library.find('sensor.camera.rgb')
    blueprint.set_attribute('fov', '110')

    sensor_spawn = carla.Transform(carla.Location(x=2.5, y=0, z=1.0))
    sensor = world.spawn_actor(blueprint, sensor_spawn, attach_to=vehicle)

    actor_list.append(sensor)
    actor_list.append(vehicle)
    
    shutil.rmtree('RGB_Collect', ignore_errors=False, onerror=None)
    os.makedirs('RGB_Collect', exist_ok=True)
    sensor.listen(analyze_image)

    time.sleep(30)

finally:
    # First, make sure all vehicles are taken off autopilot
    for actor in actor_list:
        if isinstance(actor, carla.Vehicle):
            try:
                actor.set_autopilot(False)
            except Exception as e:
                print(f"[WARNING] Couldn't disable autopilot for {actor.id}: {e}")

    for actor in actor_list:    
        actor.destroy()
    print('\nSimulation complete')
    

    
