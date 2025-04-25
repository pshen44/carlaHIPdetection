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
import json

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
        pil_image.save(buffer, format="JPEG", quality=95)  # High quality JPEG
        encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")

        captured_images.append(encoded)


gpt_prompt = """
You are operating on an autonomous vehicle equipped with a forward-facing RGB camera sensor.

Your task is to analyze a series of RGB camera images and identify the presence of High Illumination Priorities (HIPs) which include but are not limited to:
- Emergency vehicles (ambulances, police cars, fire trucks) WITH LIGHTS ON
- Cars with hazard lights FLASHING AND ON
- Any other FLASHING illuminated vehicles or objects
- Construction vehicles with warning lights
- School buses with flashing lights

You will receive a series of images and must strictly follow the JSON format, analyzing what was observed.
The response requires the following rules:

Lane Numbering Rules:
- Lanes are numbered from left to right, starting from 1
- The shoulder is counted as a lane if it's paved and accessible
- Lanes in opposing lanes are NOT counted e.g. past the double yellow lines

Car Response Rules:
- A response is required if the HIP is in the car's current road (any lane)
- No response needed for HIPs in opposing traffic (except emergency vehicles)
- ONLY For emergency vehicles, a response is required even if they're in oncoming traffic

```json
{
  "image_number": <integer, start from 1>,
  "car_lane_position": <integer, referring to rules, the current lane the agent car is in>,
  "available_lane_positions": <integer list, representing all available lanes, not including occupied lanes by other vehicles>,
  "HIPs": <string list, each string should name a HIP object, e.g., "Ambulance", "Hazard Lights", "Police Car">,
  "car_response": <boolean, true if the car should respond in any way to the HIP>
}
                     
The first three images provided ARE ALWAYS EXAMPLES AND ARE NOT PART OF THE ANALYSIS: 
    
```json
{
  "image_number": <EXAMPLE1>,
  "car_lane_position": <1>,
  "available_lane_positions": <1,2,3>,
  "HIPs": <"Ambulance">,
  "car_response": <True>
}
```json
{
  "image_number": <EXAMPLE2>,
  "car_lane_position": <2>,
  "available_lane_positions": <1,2,3>,
  "HIPs": <"Ambulance">,
  "car_response": <True>
}
```json
{
  "image_number": <EXAMPLE3>,
  "car_lane_position": <2>,
  "available_lane_positions": <1,2,3>,
  "HIPs": <"Ambulance">,
  "car_response": <True>
}

"""
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
                    {"type": "text", "text": gpt_prompt}
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
    return response_text

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
        tm.vehicle_percentage_speed_difference(ambulance, -400.0)
        actor_list.append(ambulance)

    blueprint_rgb = blueprint_library.find('sensor.camera.rgb')
    blueprint_rgb.set_attribute('fov', '110')

    sensor_spawn = carla.Transform(carla.Location(x=-0.3, y=0, z=1.7))
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
    response_text = analyze_rgb_folder("RGB_Collect", "RGB_examples") # run images through GPT
    gpt_analysis_time = time.time() - start_time
    print(f"GPT analysis took {gpt_analysis_time:.2f} seconds")
    time.sleep(6)

import csv
import os
import re

def log_to_csv(result_dict, filename="hip_log.csv"):
    file_exists = os.path.isfile(filename)
    with open(filename, mode="a", newline='') as file:
        writer = csv.DictWriter(file, fieldnames=result_dict.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(result_dict)


try:
    matches = re.findall(r"{.*?}", response_text, re.DOTALL)
    for match in matches:
        try:
            parsed = json.loads(match)
            log_to_csv(parsed)
        except Exception as jerr:
            print(f"[ERROR] Failed to parse JSON block: {jerr}")

    # Add blank row after logging all JSONs for this run
    with open("hip_log.csv", "a", newline='') as file:
        file.write(f"Analysis time: {gpt_analysis_time:.2f}")
        file.write("\n")

except Exception as e:
    print(f"[ERROR] Failed to extract/log GPT output: {e}")



