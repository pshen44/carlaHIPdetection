# High Illumination Priority (HIP) Detection Using CARLA

This project simulates an autonomous vehicle using the *CARLA simulator* (https://carla.org/) and identifies High Illumination Priority (HIP) objects in real time using the *OpenAI GPT-4.1*(https://platform.openai.com/docs/guides/vision) vision model.

HIP objects include:
- **Active** emergency vehicles
- Vehicles with hazard lights on
- Signs with flashing lights
- Snowplow trucks with hazard lights on
- etc.

Overview

The agent vehicle is equipped with an RGB camera that captures frames every few seconds. These frames are:
- Saved to disk (`RGB_Collect/`) for user reference.
The simulation then ends after a given number of seconds.
- All frames captured are sent to GPT-4o via the OpenAI API at the end of the simulation.
- Analyzed to determine the number of present HIPs, and whether the car should respond (slow down, stop, or change lanes).
- Logs all JSON outputs to hip_log.csv for data collection.

Additionally, there is a live capture script (WIP):
- Frames captured by the RGB camera are sent during the simulation for streamed responses.
- Simulation does not need to stop for frame and data collection.


===============================================

### Setup + Requirements
- Python 3.7.X (created on 3.7.16)
- CARLA 0.9.15
- OpenAI Python SDK (`openai`)
- numpy
