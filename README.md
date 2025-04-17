# High Illumination Priority (HIP) Detection Using CARLA

This project simulates an autonomous vehicle using the *CARLA simulator* (https://carla.org/) and identifies High Illumination Priority (HIP) objects in real time using the *OpenAI GPT-4o*(https://platform.openai.com/docs/guides/vision) vision model.

HIP objects include:
- **Active** emergency vehicles
- Vehicles with hazard lights on
- Signs with flashing lights
- Snowplow trucks with hazard lights on

Overview

The agent vehicle is equipped with an RGB camera that captures frames every few seconds. These frames are:
- Saved to disk (`RGB_Collect/`)
- Simultaneously sent (one at a time) to GPT-4o via the OpenAI API
- Analyzed to determine the number of present HIPs, and whether the car should respond (slow down, stop, or change lanes)

All results are printed live during simulation.

===============================================

Setup

### Requirements
- Python 3.7.X (created on 3.7.16)
- CARLA 0.9.15
- OpenAI Python SDK (`openai`)
- numpy
