from openai import OpenAI
import base64

client = OpenAI(
api_key="key"
)

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

image_path = "out/000000.png"
base64_image = encode_image(image_path)