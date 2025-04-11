import glob
import os
import random
import time
import shutil
import base64
from openai import OpenAI
import carla


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def analyze_rgb_folder(rgb_dir="RGB_Collect", example_dir="RGB_examples"):
    client = OpenAI()
    image_files = sorted(glob.glob(f"{rgb_dir}/*.png"))
    example_image_files = sorted(glob.glob(f"{example_dir}/*.png"))

    base64_examples = [encode_image(img) for img in example_image_files]
    base64_images = [encode_image(img) for img in image_files]

    if len(base64_images) < 5:
        raise ValueError("Not enough images found. Ensure there are at least 5 images in the directory.")

    completion = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": """
You are equipped on an autonomous vehicle with an RGB camera sensor.
You are tasked with detecting high illumination priority (HIP) objects, from the RGB camera sensor.
A high illumination priority (HIP) is an object, car, and/or obstacle that is characterized as
having bright lights and is a priority while driving to recognize.
Ambulances are classified as HIPs.
Cars with hazard lights flashing are HIPs.
State the parameters of the images clearly in JSON format.
Take note of road position, any nearby obstacles, and especially any HIPs.
HIPs NOT in the current road the car is driving on do NOT require a response.
You will be given lots of images, and you must pick the best ones that expose the most HIPs.
ALWAYS follow this JSON format:

    image_number: []
    car_lane_position: [use lane numbers in order of increasing from left to right]
    available_lane_positions: [list of lane positions in order of increasing from left to right]
    HIP_count: [# of HIPs] 
    HIPs: [list of HIPs] 
    HIP_positions: [compass direction of HIP with respect to car] 
    car_response: [true/false]
    
An example image and correct JSON response is provided. This image is to be ignored in the analysis.
    
    image_number: 1
    car_lane_position: 2
    available_lane_positions: [1, 2, 3]
    HIPs: Ambulance,
    HIP_positions: [NorthWest]
    car_response: false
                    """},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_examples[0]}"}},
                ] + [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img}"}} for img in base64_images
                ]
            }
        ]
    )

    
    print(completion.choices[0].message.content)

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
    sensor.listen(lambda image: (image.save_to_disk('RGB_Collect/%06d.png' % image.frame)
                                 if image.frame % 60 == 0 else None))

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
