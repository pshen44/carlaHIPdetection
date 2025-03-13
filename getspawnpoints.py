import carla

# Connect to the CARLA server (replace with your server address and port)
client = carla.Client('localhost', 2000)
client.set_timeout(10.0)

# Get the world
world = client.get_world()

# Get the map
map = world.get_map()

# Get the spawn points
spawn_points = map.get_spawn_points()

# Print the number of spawn points
print(f"Number of spawn points: {len(spawn_points)}")

# You can now iterate through the spawn_points and spawn actors at these locations
# Example:
# for spawn_point in spawn_points:
#     vehicle_blueprint = world.get_blueprint_library().filter("vehicle.*")[0]
#     vehicle = world.spawn_actor(vehicle_blueprint, spawn_point)