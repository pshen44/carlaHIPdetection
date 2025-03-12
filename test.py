class CarEnv: # create class
    """This is the Car Environment that we are
    defining"""
    SHOW_CAM = SHOW_PREVIEW
    STEER_AMT = 1.0 # steer left
    im_width = IM_WIDTH
    im_height = IM_HEIGHT
    front_camera = None

    def __init__(self): # create method init
        self.client = carla.Client # assign fields to object
        self.client = carla.Client('localhost', 2000)
        self.client.set_timeout(2.0)

        self.world = client.get_world()

        self.blueprint_library = world.get_blueprint_library()
        self.blueprint_library.filter('model3')[0]
    def reset_car(self):
        """reset car"""
        self.collision_hist = [] # any collision will be added to vec
        self.actor_list =[] # track actors as well
        
        self.transform = random.choice(self.world.get_map().get.spawn_points())
        self.vehicle = self.world.spawn_actor(self.model_3, self.transform)

        self.rgb_cam = self.blueprint_library.find('sensor.camera.rgb')
        self.rgb_settings("image_size_x", f"{self.im_width}")
        self.rgb_settings("image_size_y", f"{self.im_height}")
        self.rgb_settings("field_of_view", f"120")

        transform = carla.transform(carla.Location(x=2.5, z=0.7))
