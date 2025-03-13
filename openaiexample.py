from openai import OpenAI
import base64
import os
import glob # for getting latest out/000000.png

client = OpenAI(
api_key="key"
)

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

image_files = sorted(glob.glob("out/*.png"))
if image_files:
    image_path = image_files[-1]  # Get the latest image
else:
    raise ("No images found in the out/ folder.")

image_path = "out/000000.png"
base64_image = encode_image(image_path)

completion = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {
            "role": "user",
            "content": [
                { "type": "text", "text": "State the parameters of this image clearly."
                "Use up to three sentences."
                "Take note of road position, nearby obstacles, and any anomalies in the last sentence." },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}",
                    },
                },
            ],
        }
    ],
)

print(completion.choices[0].message.content)