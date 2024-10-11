INF = 1000000

class BaseAbility():
    def __init__(self):
        self.REACH_FOR_THRESHOLD = 3.0
        # The maximal distance to sucessfully pick up and put on
        self.LOWEST_PICKUP_HEIGHT = 0
        self.HIGHEST_PICKUP_HEIGHT = 3
        self.LOWEST_PUT_ON_HEIGHT = 0
        self.HIGHEST_PUT_ON_HEIGHT = 3
        self.HIGHEST_PICKUP_MASS = 600
        self.WHEELCHAIRED = False
        self.RIDING = False

class HelperAbility(BaseAbility):
    def __init__(self):
        super().__init__()

class GirlAbility(BaseAbility):
    def __init__(self):
        super().__init__()
        self.HIGHEST_PICKUP_HEIGHT = 1.5
        self.HIGHEST_PUT_ON_HEIGHT = 1.5
        self.HIGHEST_PICKUP_MASS = 100

class OldManAbility(BaseAbility):
    def __init__(self):
        super().__init__()
        self.LOWEST_PICKUP_HEIGHT = 0.25
        self.LOWEST_PUT_ON_HEIGHT = 0.25

class WheelchairAbility(BaseAbility):
    def __init__(self):
        super().__init__()
        self.WHEELCHAIRED = True
        self.LOWEST_PICKUP_HEIGHT = 0.25
        self.LOWEST_PUT_ON_HEIGHT = 0.25
        self.HIGHEST_PICKUP_HEIGHT = 1.5
        self.HIGHEST_PUT_ON_HEIGHT = 1.5
        self.HIGHEST_PICKUP_MASS = 500

class RiderAbility(BaseAbility):
    def __init__(self):
        super().__init__()
        self.RIDING = True

class WomanAbility(BaseAbility):
    def __init__(self):
        super().__init__()
        self.HIGHEST_PICKUP_MASS = 100

ability_mapping = {
    'helper': HelperAbility,
    'girl': GirlAbility,
    'wheelchair': WheelchairAbility,
    'old': OldManAbility,
    'rider': RiderAbility,
    'woman': WomanAbility
}