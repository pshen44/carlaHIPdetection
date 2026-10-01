from openai import OpenAI
import base64
import glob # for getting latest out/000000.png
# api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI()


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")
    

# image_path = "C:/Users/shenx/Desktop/CARLA/CARLA_0.9.15/WindowsNoEditor/GITHUB_PULL/carlaHIPdetection/RGB_Collect/"
image_files = sorted(glob.glob("C:/Users/shenx/Desktop/CARLA/CARLA_0.9.15/WindowsNoEditor/GITHUB_PULL/carlaHIPdetection/RGB_Collect/*.png"))

# example_image_path = "C:/Users/shenx/Desktop/CARLA/CARLA_0.9.15/WindowsNoEditor/GITHUB_PULL/carlaHIPdetection/RGB_Collect/"
example_image_files = sorted(glob.glob("C:/Users/shenx/Desktop/CARLA/CARLA_0.9.15/WindowsNoEditor/GITHUB_PULL/carlaHIPdetection/RGB_examples/*.png"))

base64_examples = [encode_image(image_file) for image_file in example_image_files]
base64_images = [encode_image(image_file) for image_file in image_files]




completion = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {
            "role": "user",
            "content": [
                { "type": "text", "text": """
                 
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
                HIP_count: [# of HIPs] 
                HIPs: [list of HIPs] 
                HIP_positions: [list of HIP positions respectively in 'lane_position' format - An HIP NOT in the current road as the car would be 'N/A'] 
                car_response: [true/false]
                 
                An example image and correct JSON response is provided. This image is to be ignored in the analysis.
                 
                image_number: 1 
                car_lane_position: 2 
                HIP_count: 1
                HIPs: Ambulance,
                HIP_positions: N/A
                car_response: false

                """
                },
                *[
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_examples[0]}",
                        },
                    }
                ]
                +[
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_images[0]}",
                        },
                    } #for base64_image in base64_images
                ]
            ],
        }
    ],
)

print(completion.choices[0].message.content)