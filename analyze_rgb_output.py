from openai import OpenAI
import base64
import os
import glob # for getting latest out/000000.png

client = OpenAI(
api_key="key"
)
image_path = "out/000000.png"
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

image_files = sorted(glob.glob("out/*.png"))
if image_files:
    if len(image_files) >= 5:
        image_path = image_files[4]  # Get the 5th image
    else:
        raise ValueError("There are less than 5 images in the out/ folder.")
else:
    raise ("No images found in the out/ folder.")

base64_images = [encode_image(image_file) for image_file in image_files]


# base64_image = encode_image(image_path)

completion = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {
            "role": "user",
            "content": [
                { "type": "text", "text": 
                "You are equipped on a autonomous vehicle with an RGB camera sensor."
                "A high illumination priority (HIP) is an object, car, and/or obstacle that is characterized as"
                "having bright lights and is a priority while driving to recognize."
                "Ambulances are classified as HIPs."
                "Drivers should yield to ambulances and give them the right of way."
                "State the parameters of the two images clearly."
                "Use up to five sentences."
                "Take note of road position, any nearby obstacles, and especially any HIPs."
                "Point out differences between the two images."
                "In the final sentence, briefly explain how the vehicle should behave in the scenario." 
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_images[0]}",
                    },
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_images[5]}",
                    },
                }
                
                
            ],
        }
    ],
)

print(completion.choices[0].message.content)