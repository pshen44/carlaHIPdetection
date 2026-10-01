"""The prompt and output schema shared by every model backend.

Compared with the old prompts this one:
* states a single, consistent response rule (the old analysis prompt said
  "respond to any HIP", the live one said "only on our road");
* states the same lane-numbering rule the ground truth uses;
* asks for strict JSON matching ``OUTPUT_SCHEMA`` (no angle-bracket
  pseudo-JSON examples, no regex scraping of the reply);
* tells the model when it is looking at a time-ordered burst, so it can use
  flashing as evidence.
"""

SYSTEM_PROMPT = """You are the perception module of an autonomous car with one forward-facing RGB camera.

Your job is to detect High Illumination Priority objects (HIPs). A HIP is:
- an emergency vehicle (ambulance, police car, fire truck) whose emergency lights are ON, or
- any vehicle with its hazard lights (both turn signals) ON.
An emergency vehicle with its emergency lights OFF is NOT a HIP. Ordinary headlights, tail lights,
brake lights, street lights and traffic lights are NOT HIPs.

The car must respond (slow down, stop or change lanes) if a HIP is
- in the car's own lane or another lane travelling in the same direction, or
- an emergency vehicle in the oncoming lanes,
and it is within about 80 metres.

Lane numbering: count only lanes travelling in the same direction as the car, from left to right,
starting at 1. Do not count oncoming lanes or shoulders.

Reply with a single JSON object and nothing else:
{
  "hip_present": true or false,
  "hips": [ {"type": "ambulance" | "police" | "firetruck" | "hazard_vehicle" | "other",
             "relation": "same_lane" | "adjacent_lane" | "oncoming" | "other"} ],
  "car_response": true or false,
  "ego_lane": integer,
  "n_lanes": integer,
  "confidence": number between 0 and 1 that a HIP is present
}
"""

SINGLE_FRAME_NOTE = "You are given one camera frame."
BURST_NOTE = ("You are given {n} consecutive camera frames from the same moment, {dt:.2f} s apart, "
              "in time order. Lights that change between frames are flashing.")

OUTPUT_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["hip_present", "hips", "car_response", "ego_lane", "n_lanes", "confidence"],
    "properties": {
        "hip_present": {"type": "boolean"},
        "hips": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["type", "relation"],
                "properties": {
                    "type": {"type": "string",
                             "enum": ["ambulance", "police", "firetruck", "hazard_vehicle", "other"]},
                    "relation": {"type": "string",
                                 "enum": ["same_lane", "adjacent_lane", "oncoming", "other"]},
                },
            },
        },
        "car_response": {"type": "boolean"},
        "ego_lane": {"type": "integer"},
        "n_lanes": {"type": "integer"},
        "confidence": {"type": "number"},
    },
}


def user_text(n_frames: int, dt: float) -> str:
    if n_frames <= 1:
        return SINGLE_FRAME_NOTE
    return BURST_NOTE.format(n=n_frames, dt=dt)
