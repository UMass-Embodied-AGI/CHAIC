import random

class PlanAgent:
    def __init__(self, id, args):
        print(id)
        print(args)
        pass

    def reset(self, obs, info):
        pass

    def act(self, obs):
        # useful objects in obs:
        # "rgb": RGB image of the current agent's view
        # "depth": depth image of the current agent's view
        # "camera_matrix": the camera matrix of current agent's ego camera
        # "FOV": the field of view of current agent's ego camera
        # "agent": a list of length 6 that contains the position (x,y,z) and forward (fx,fy,fz) of the agent, formatted as [x, y, z, fx, fy, fz]. 
        # "held_obejcts": all the objects that current agent is holding. 
        # It is a list of length 2 that contains the information of the object that is held in the agent's two hands. 
        # Each object's information contains its name, type and a unique id. 
        # If it's a container, it also includes the information of the objects in it
        # "status": the status of current action, which is a number from 0 to 2. 0 for 'ongoing', 1 for 'failure', 2 for 'success'.
        # "current_frames": the number of frames passed
        # "valid": whether the last action of the agent is valid
        # "previous_action" & "previous_status": all previous actions of the agent and their corresponding status
        
        t = random.randint(0, 2)
        return {"type": t}