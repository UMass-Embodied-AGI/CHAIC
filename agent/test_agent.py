import random

class PlanAgent:
    def __init__(self, id, args):
        pass

    def reset(obs, info):
        print("obs: ", obs)
        print("info: ", info)

    def act(obs):
        t = random.randint(0, 3)
        return {"type": t}