from openai import OpenAI
import base64
import glob # for getting latest out/000000.png
# api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI()


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")
    

image_path = "C:/Users/shenx/Desktop/CARLA/CARLA_0.9.15/WindowsNoEditor/GITHUB_PULL/carlaHIPdetection/RGB_Collect/"
image_files = sorted(glob.glob("C:/Users/shenx/Desktop/CARLA/CARLA_0.9.15/WindowsNoEditor/GITHUB_PULL/carlaHIPdetection/RGB_Collect/*.png"))


base64_images = [encode_image(image_file) for image_file in image_files]

# Ensure there are at least 8 images
if len(base64_images) < 5:
    raise ValueError("Not enough images found. Ensure there are at least 8 images in the directory.")


# base64_image = encode_image(image_path)

completion = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {
            "role": "user",
            "content": [
                { "type": "text", "text": 
                "You are equipped on an autonomous vehicle with an RGB camera sensor."
                "A high illumination priority (HIP) is an object, car, and/or obstacle that is characterized as"
                "having bright lights and is a priority while driving to recognize."
                "Ambulances are classified as HIPs."
                "Cars with hazard lights flashing are HIPs."
                "State the parameters of the images clearly."
                "Use up to seven sentences."
                "Take note of road position, any nearby obstacles, and especially any HIPs."
                "Point out differences between the images, and state how many there are."
                "In the final sentence, briefly explain how the vehicle should behave in the scenario." 
                },
                *[
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}",
                        },
                    } for base64_image in base64_images
                ]
            ],
        }
    ],
)

print(completion.choices[0].message.content)