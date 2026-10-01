# HIP benchmark results

Dataset: `data/benchmark`: 442 samples, 170 with a visible HIP, 272 without.

## HIP detection

| predictor | n | accuracy | precision | recall | F1 | FPR | AUC | response acc | response F1 | ego-lane acc | parse fail |
|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline_color_single | 221 | 47.5 | 34.2 | 28.3 | 31.0 | 38.8 | 0.46 | – | – | – | 0 |
| baseline_flicker_burst | 221 | 61.5 | 58.1 | 27.2 | 37.0 | 14.0 | 0.57 | – | – | – | 0 |
| claude-haiku__burst | 442 | 63.1 | 55.6 | 20.6 | 30.0 | 10.3 | 0.43 | 65.8 | 31.7 | 52.0 | 0 |
| claude-haiku__single | 442 | 63.1 | 55.6 | 20.6 | 30.0 | 10.3 | 0.54 | 65.8 | 31.7 | 48.0 | 0 |
| claude-opus__burst | 442 | 75.3 | 78.5 | 49.4 | 60.6 | 8.5 | 0.70 | 78.1 | 62.5 | 99.3 | 0 |
| claude-opus__single | 442 | 65.4 | 62.3 | 25.3 | 36.0 | 9.6 | 0.62 | 68.1 | 37.3 | 97.5 | 0 |
| claude-sonnet__burst | 442 | 70.6 | 76.3 | 34.1 | 47.2 | 6.6 | 0.65 | 73.1 | 48.9 | 77.6 | 0 |
| claude-sonnet__single | 442 | 64.0 | 60.4 | 18.8 | 28.7 | 7.7 | 0.58 | 66.3 | 29.4 | 53.2 | 0 |
| claude-sonnet__single_swapRB | 442 | 64.5 | 62.7 | 18.8 | 29.0 | 7.0 | 0.61 | 67.2 | 30.6 | 58.4 | 0 |

## Recall on visible HIPs by weather

| predictor | ClearNight | ClearNoon | ClearSunset | HardRainNight |
|---|---|---|---|---|
| baseline_color_single | 10.0 (n=20) | 35.0 (n=20) | 61.5 (n=26) | 3.8 (n=26) |
| baseline_flicker_burst | 35.0 (n=20) | 15.0 (n=20) | 26.9 (n=26) | 30.8 (n=26) |
| claude-haiku__burst | 30.2 (n=43) | 17.1 (n=41) | 7.1 (n=42) | 27.3 (n=44) |
| claude-haiku__single | 37.2 (n=43) | 12.2 (n=41) | 16.7 (n=42) | 15.9 (n=44) |
| claude-opus__burst | 53.5 (n=43) | 46.3 (n=41) | 42.9 (n=42) | 54.5 (n=44) |
| claude-opus__single | 37.2 (n=43) | 17.1 (n=41) | 14.3 (n=42) | 31.8 (n=44) |
| claude-sonnet__burst | 46.5 (n=43) | 29.3 (n=41) | 23.8 (n=42) | 36.4 (n=44) |
| claude-sonnet__single | 25.6 (n=43) | 17.1 (n=41) | 14.3 (n=42) | 18.2 (n=44) |
| claude-sonnet__single_swapRB | 23.3 (n=43) | 12.2 (n=41) | 16.7 (n=42) | 22.7 (n=44) |

## Recall on visible HIPs by distance

| predictor | 20-40m | 40-65m | <=20m | >65m |
|---|---|---|---|---|
| baseline_color_single | 30.8 (n=26) | 41.7 (n=24) | 13.6 (n=22) | 25.0 (n=20) |
| baseline_flicker_burst | 38.5 (n=26) | 20.8 (n=24) | 36.4 (n=22) | 10.0 (n=20) |
| claude-haiku__burst | 22.7 (n=44) | 7.1 (n=42) | 41.7 (n=48) | 5.6 (n=36) |
| claude-haiku__single | 31.8 (n=44) | 2.4 (n=42) | 33.3 (n=48) | 11.1 (n=36) |
| claude-opus__burst | 61.4 (n=44) | 31.0 (n=42) | 75.0 (n=48) | 22.2 (n=36) |
| claude-opus__single | 36.4 (n=44) | 26.2 (n=42) | 29.2 (n=48) | 5.6 (n=36) |
| claude-sonnet__burst | 43.2 (n=44) | 21.4 (n=42) | 62.5 (n=48) | 0.0 (n=36) |
| claude-sonnet__single | 27.3 (n=44) | 7.1 (n=42) | 29.2 (n=48) | 8.3 (n=36) |
| claude-sonnet__single_swapRB | 22.7 (n=44) | 11.9 (n=42) | 31.2 (n=48) | 5.6 (n=36) |

## Recall on visible HIPs by hip_type

| predictor | ambulance | firetruck | hazard_car | police |
|---|---|---|---|---|
| baseline_color_single | 16.0 (n=25) | 42.3 (n=26) | 27.8 (n=18) | 26.1 (n=23) |
| baseline_flicker_burst | 24.0 (n=25) | 53.8 (n=26) | 5.6 (n=18) | 17.4 (n=23) |
| claude-haiku__burst | 22.7 (n=44) | 35.6 (n=45) | 2.6 (n=39) | 19.0 (n=42) |
| claude-haiku__single | 27.3 (n=44) | 42.2 (n=45) | 2.6 (n=39) | 7.1 (n=42) |
| claude-opus__burst | 65.9 (n=44) | 66.7 (n=45) | 15.4 (n=39) | 45.2 (n=42) |
| claude-opus__single | 47.7 (n=44) | 37.8 (n=45) | 0.0 (n=39) | 11.9 (n=42) |
| claude-sonnet__burst | 45.5 (n=44) | 51.1 (n=45) | 2.6 (n=39) | 33.3 (n=42) |
| claude-sonnet__single | 34.1 (n=44) | 26.7 (n=45) | 2.6 (n=39) | 9.5 (n=42) |
| claude-sonnet__single_swapRB | 36.4 (n=44) | 28.9 (n=45) | 2.6 (n=39) | 4.8 (n=42) |

## Recall on visible HIPs by placement

| predictor | adjacent_lane | oncoming | same_lane |
|---|---|---|---|
| baseline_color_single | 26.5 (n=34) | 37.5 (n=32) | 19.2 (n=26) |
| baseline_flicker_burst | 29.4 (n=34) | 21.9 (n=32) | 30.8 (n=26) |
| claude-haiku__burst | 23.0 (n=61) | 22.2 (n=54) | 16.4 (n=55) |
| claude-haiku__single | 19.7 (n=61) | 22.2 (n=54) | 20.0 (n=55) |
| claude-opus__burst | 47.5 (n=61) | 57.4 (n=54) | 43.6 (n=55) |
| claude-opus__single | 19.7 (n=61) | 29.6 (n=54) | 27.3 (n=55) |
| claude-sonnet__burst | 21.3 (n=61) | 48.1 (n=54) | 34.5 (n=55) |
| claude-sonnet__single | 6.6 (n=61) | 24.1 (n=54) | 27.3 (n=55) |
| claude-sonnet__single_swapRB | 16.4 (n=61) | 20.4 (n=54) | 20.0 (n=55) |

## False-positive rate on negatives

| predictor | ambulance_lights_off | firetruck_lights_off | hazard_car_lights_off | no_target | not_visible | police_lights_off |
|---|---|---|---|---|---|---|
| baseline_color_single | 40.9 (n=22) | 61.1 (n=18) | 28.6 (n=21) | 45.8 (n=24) | 40.0 (n=20) | 20.8 (n=24) |
| baseline_flicker_burst | 4.5 (n=22) | 55.6 (n=18) | 4.8 (n=21) | 12.5 (n=24) | 15.0 (n=20) | 0.0 (n=24) |
| claude-haiku__burst | 19.1 (n=47) | 30.2 (n=43) | 2.4 (n=41) | 5.2 (n=58) | 4.8 (n=42) | 0.0 (n=41) |
| claude-haiku__single | 14.9 (n=47) | 20.9 (n=43) | 2.4 (n=41) | 8.6 (n=58) | 9.5 (n=42) | 4.9 (n=41) |
| claude-opus__burst | 17.0 (n=47) | 16.3 (n=43) | 12.2 (n=41) | 1.7 (n=58) | 4.8 (n=42) | 0.0 (n=41) |
| claude-opus__single | 19.1 (n=47) | 16.3 (n=43) | 4.9 (n=41) | 6.9 (n=58) | 7.1 (n=42) | 2.4 (n=41) |
| claude-sonnet__burst | 17.0 (n=47) | 23.3 (n=43) | 0.0 (n=41) | 0.0 (n=58) | 0.0 (n=42) | 0.0 (n=41) |
| claude-sonnet__single | 14.9 (n=47) | 25.6 (n=43) | 0.0 (n=41) | 3.4 (n=58) | 2.4 (n=42) | 0.0 (n=41) |
| claude-sonnet__single_swapRB | 19.1 (n=47) | 18.6 (n=43) | 0.0 (n=41) | 0.0 (n=58) | 2.4 (n=42) | 2.4 (n=41) |
