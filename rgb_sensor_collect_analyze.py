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

captured_images = []
def handle_image(image):
    if image.frame % 60 == 0:
        # Save to disk
        image.save_to_disk(f"RGB_Collect/{image.frame:06d}.png")

        # Convert CARLA raw data to RGB image
        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        array = np.reshape(array, (image.height, image.width, 4))[:, :, :3]

        pil_image = Image.fromarray(array)  
        buffer = io.BytesIO()
        pil_image.save(buffer, format="JPEG")
        encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")

        captured_images.append(encoded)


def analyze_rgb_folder(rgb_dir="RGB_Collect", example_dir="RGB_examples"):
    client = OpenAI()
    example_image_files = sorted(glob.glob(f"{example_dir}/*.png"))
    base64_examples = [encode_image(img) for img in example_image_files]

    completion = client.chat.completions.create(
        model="gpt-4.1",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": """
You are operating on an autonomous vehicle equipped with a forward-facing RGB camera sensor.

Your task is to analyze a series of RGB camera images and identify the presence of High Illumination Priority (HIP) objects — any object emitting strong lights that must be prioritized by the vehicle. These include:
- Emergency vehicles only WITH LIGHTS ON
- Cars with hazard lights FLASHING AND ON
- Any other FLASHING illuminated vehicles

You will receive a series of images and must respond with **one JSON object per image**, summarizing what was observed.
                     
Car response to HIPs is defined as any action that would require the car to slow down, stop, or change lanes.
A response will only be required if the HIP is in the car's current road (any lane), or if the HIP is in oncoming traffic ONLY for emergency vehicles.

Each JSON object must strictly follow this format:

```json
{
  "image_number": <integer, the number shown in the image filename - start from 1>,
  "car_lane_position": <integer, representing the lane the car is currently in, not counting lanes with opposing traffic. Lane numbers increase from left to right [The shoulder is counted]>,
  "available_lane_positions": <list of integers, representing all lanes currently visible on the road from left to right>,
  "HIPs": <list of strings, each string should name a HIP object, e.g., "Ambulance", "Hazard Lights", "Flashing Vehicle">,
  "HIP_positions": <list of strings, each string describing the compass direction (e.g., "North", "NorthWest", "East") of the HIPs with respect to the car>,
  "car_response": <boolean, true if the car should slow down, stop, or change lanes due to the HIP; false otherwise>
}
                     
The first three images provided ARE ALWAYS EXAMPLES AND ARE NOT PART OF THE ANALYSIS: 
                                  
```json
{
  "image_number": <EXAMPLE1>,
  "car_lane_position": <2>,
  "available_lane_positions": <1,2,3>,
  "HIPs": <"Ambulance">,
  "HIP_positions": <"Northwest">,
  "car_response": <True>
}

```json
{
  "image_number": <EXAMPLE2>
  "car_lane_position": <1>,
  "available_lane_positions": <1,2>,
  "HIPs": <>,
  "HIP_positions": <>,
  "car_response": <False>
}

```json
{
  "image_number": <EXAMPLE3>,
  "car_lane_position": <2>,
  "available_lane_positions": <1,2,3>,
  "HIPs": <"Ambulance">,
  "HIP_positions": <"Northwest">,
  "car_response": <True>
}                     
                     

                    """}
] + [
    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{examples}"}} for examples in base64_examples
] + [
    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img}"}} for img in captured_images
]
            }
        ]
    )

    
    response_text = completion.choices[0].message.content
    print(response_text)

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

    blueprint_rgb = blueprint_library.find('sensor.camera.rgb')
    blueprint_rgb.set_attribute('fov', '110')

    sensor_spawn = carla.Transform(carla.Location(x=2.5, y=0, z=1.0))
    sensor_rgb = world.spawn_actor(blueprint_rgb, sensor_spawn, attach_to=vehicle)

    actor_list.append(sensor_rgb)
    actor_list.append(vehicle)
    
    shutil.rmtree('RGB_Collect', ignore_errors=False, onerror=None)
    os.makedirs('RGB_Collect', exist_ok=True)
    sensor_rgb.listen(handle_image)

    time.sleep(10)

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
    print('Simulation complete')
    
    start_time = time.time()
    analyze_rgb_folder("RGB_Collect", "RGB_examples")
    print(f"GPT analysis took {time.time() - start_time:.2f} seconds")
    time.sleep(6)

    
