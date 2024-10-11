from tdw.controller import Controller
from tdw.add_ons.proc_gen_kitchen import ProcGenKitchen
from tdw.add_ons.third_person_camera import ThirdPersonCamera
from tdw.add_ons.image_capture import ImageCapture
from tdw.add_ons.floorplan import Floorplan
from tdw.tdw_utils import TDWUtils
from tdw.librarian import SceneLibrarian, ModelLibrarian
from tdw.add_ons.object_manager import ObjectManager
from tdw.add_ons.occupancy_map import OccupancyMap
from tdw.scene_data.scene_bounds import SceneBounds
from tdw.container_data.container_tag import ContainerTag
from tdw.add_ons.container_manager import ContainerManager
from utils import belongs_to_which_room, shift, get_room_functional_by_id
import random
import json
import numpy as np
from collections import Counter

import sys
import os


#scene settings
settings = sys.argv[1:]
FLOORPLAN_SCENE_NAME = settings[0]
FLOORPLAN_LAYOUT = int(settings[1])
scene_id = int(settings[2])
dataset_prefix = settings[3]
print("FLOORPLAN_SCENE_NAME:", FLOORPLAN_SCENE_NAME)
print("FLOORPLAN_LAYOUT:", FLOORPLAN_LAYOUT)
print("scene_id:", scene_id)
print()

# container_limit, container_room_limit = 0, 0
# if scene_id == 0: container_limit, container_room_limit = 5, 2
# if scene_id == 1: container_limit, container_room_limit = 2, 1
# if scene_id == 2: container_limit, container_room_limit = 4, 1

#no lowcontainer now
possible_keyword = ['normal', 'lowthing', 'highthing', 'highcontainer', 'highgoalplace', 'wheelchair']
constraint_map = {'normal' : None, 'lowthing' : 'low', 'highthing' : 'high', 'lowcontainer' : 'low', 'highcontainer' : 'high', 'highgoalplace' : 'high', 'wheelchair' : 'wheelchair'}
keyword = dataset_prefix
constraint_type = constraint_map[keyword]

with open('./dataset/list.json', 'r') as f:
    object_place = json.loads(f.read())

with open('./dataset/name_map.json', 'r') as f:
    name_map = json.loads(f.read())

with open('./dataset/object_scale.json', 'r') as f:
    object_scale = json.loads(f.read())


def generate_balanced_list(a):
    if not 1 <= len(a) <= 5:
        raise ValueError("The input list must contain 1 to 5 items.")

    # Shuffle the input list to use a randomized copy
    a_copy = a[:]
    random.shuffle(a_copy)

    # Determine the length of the return list
    list_length = len(a) + 4

    # Initialize the counter to distribute items as evenly as possible
    count = Counter()
    while sum(count.values()) < list_length:
        for item in a_copy:
            count[item] += 1
            if sum(count.values()) >= list_length:
                break

    # Convert the counter to a list of tuples
    balanced_list = [(item, cnt) for item, cnt in count.items()]

    return balanced_list

def combine_and_randomize_lists(list1, list2, list3, list4):
    # Generate balanced lists for each input list
    balanced_list1 = generate_balanced_list(list1)
    balanced_list2 = generate_balanced_list(list2)
    balanced_list3 = generate_balanced_list(list3)
    balanced_list4 = generate_balanced_list(list4)
    
    # Combine all balanced lists and shuffle the result
    combined_list = balanced_list1 + balanced_list2 + balanced_list3 + balanced_list4
    random.shuffle(combined_list)
    
    def generate_task():
        # Randomly select one of the balanced lists
        chosen_list = random.choice([balanced_list1, balanced_list2, balanced_list3, balanced_list4])
        
        # Randomly select an item from one of the other lists
        other_lists = [lst for lst in [balanced_list1, balanced_list2, balanced_list3, balanced_list4] if lst != chosen_list]
        additional_item = random.choice(random.choice(other_lists))
        
        # Add the additional item to the chosen list and shuffle
        combined_with_additional = chosen_list + [(additional_item[0], 1)]
        if keyword == 'highthing':
            combined_with_additional += [('vase_05', 2)]
        random.shuffle(combined_with_additional)
        return combined_with_additional
    
    return combined_list, generate_task

all_objects_need_to_put, generate_task = combine_and_randomize_lists(object_place['food']['target_fruit'], object_place['food']['target_bread'], object_place['stuff']['work'], object_place['stuff']['school'])
task = generate_task()

#expand the num, like (apple, 2) -> [apple, apple]
remaining_objects = []
for i in all_objects_need_to_put:
    remaining_objects += [i[0]] * i[1]
np.random.shuffle(remaining_objects)
if keyword == 'lowthing':
    # split half to remaining_low_objects
    remaining_objects, remaining_low_objects = remaining_objects[:len(remaining_objects)//2], remaining_objects[len(remaining_objects)//2:]

# Here should has a if keyword == 'highthing', but because it needs highplace num, I move the code down.

all_containers_need_to_put = [(i,1) for i in object_place['food']['container'] + object_place['stuff']['container']]
print("all_containers_need_to_put ", all_containers_need_to_put)
print()

remaining_containers = []
for i in all_containers_need_to_put:
    remaining_containers += [i[0]] * i[1]

np.random.shuffle(remaining_containers)

# for the highthing part, this line is in the next part of code, after the high_place_pointer is generated.
if keyword not in ['lowcontainer', 'highcontainer', 'highthing']:
    remaining_objects += remaining_containers

np.random.shuffle(remaining_objects)

def check_map(x, z, occ_map, positions, threshold = 0.65):
    for i in range(occ_map.shape[0]):
        for j in range(occ_map.shape[1]):
            if occ_map[i, j] == 1 and (positions[i, j][0] - positions[x, z][0]) ** 2 + (positions[i, j][1] - positions[x, z][1]) ** 2 < threshold ** 2:
                return False
    return True

#start simulator
port = 1087
os.system(f"pkill -f 'TDW.x86_64 -port {port}'")
#rm dataset?
c = Controller(port = port)


floorplan = Floorplan()

floorplan.init_scene(scene=FLOORPLAN_SCENE_NAME, layout=FLOORPLAN_LAYOUT)

occ = OccupancyMap()
occ.generate(cell_size=0.25, once=False)
om = ObjectManager(transforms=True, rigidbodies=True, bounds=True)
container_manager = ContainerManager()
c.add_ons.extend([floorplan, occ, om, container_manager])
# Create the scene.
c.communicate([])
# Finish initializing om
c.communicate([])
c.communicate([])
c.communicate([])

os.makedirs("./dataset/", exist_ok=True)
os.makedirs(f"./dataset/{dataset_prefix}/", exist_ok=True)

commands_init_scene = [{"$type": "set_screen_size", "width": 1920, "height": 1080}] # Set screen size
commands_init_scene.extend(floorplan.commands)
commands_init_scene.append({"$type": "set_floorplan_roof", "show": False}) # Hide roof

response = c.communicate(commands_init_scene)

resp = c.communicate([{"$type": "send_scene_regions"}])
scene_bounds = SceneBounds(resp=resp)

commands = []
object_room_list = []
camera = ThirdPersonCamera(position={"x": 0, "y": 30, "z": 0},
                           look_at={"x": 0, "y": 0, "z": 0},
                           avatar_id="a")

screenshot_path = f"./dataset/{dataset_prefix}/screenshots/{FLOORPLAN_SCENE_NAME}_{FLOORPLAN_LAYOUT}_{scene_id}"
if not os.path.exists(screenshot_path):
    os.makedirs(screenshot_path)
print(f"Images will be saved to: {screenshot_path}")

capture = ImageCapture(avatar_ids=["a"], path=screenshot_path, pass_masks=["_img"])
commands.extend(camera.get_initialization_commands()) # Init camera
c.add_ons.extend([capture]) # Capture images

def get_scale(obj_name, object_scale):
    if obj_name in object_scale:
        return 0.8*object_scale[obj_name]
    else:
        print("WARNING: No scale for object: ", obj_name)
        return 0.8

object_list = []
container_list = []

#only contain objects or containers that is "put on" something, or say not on floor.
place_record = {}

place_list = []
for object_id in om.objects_static:
    place_list.append(object_id)
np.random.shuffle(place_list)

#place_pointer: (object_id, i, func_name, id). The first two can define it.


all_place_pointer = []
high_place_pointer = []
high_position_name = []

for object_id in place_list:
    id = belongs_to_which_room(om.bounds[object_id].top[0], om.bounds[object_id].top[2], scene_bounds)
    func_name = get_room_functional_by_id(FLOORPLAN_SCENE_NAME, FLOORPLAN_LAYOUT, id)
    if id == -1: continue
    if om.objects_static[object_id].category in ['table', 'sofa', 'chair']: # place objects on specific objects
        if om.objects_static[object_id].category in ['table']:
            num = 3
        else:
            num = 2
        for i in range(num):
            all_place_pointer.append((object_id, i, func_name, id))
    # is this 1.0 or 1.5?
    if shift(om.bounds[object_id], 0)[1] > 1.5 and om.objects_static[object_id].category in ['refrigerator', 'cabinet'] and 'door' not in om.objects_static[object_id].name:
        high_position_name.append(om.objects_static[object_id].category)
        if keyword in ['highthing', 'highcontainer']:
            num = 2
            if keyword == "highcontainer":
                num = 1
            for i in range(num):
                high_place_pointer.append((object_id, i, func_name, id))

def place(obj_name, place_pointer):
    object_id, i, func_name, id = place_pointer
    obj = c.get_unique_id()
    p = shift(om.bounds[object_id], i)
    object_room_list.append([obj_name, func_name])
    commands.extend(c.get_add_physics_object(obj_name,
                            object_id=obj,
                            position=TDWUtils.array_to_vector3(p),
                            scale_factor = {
                                "x": get_scale(obj_name, object_scale),
                                "y": get_scale(obj_name, object_scale),
                                "z": get_scale(obj_name, object_scale),
                            }))
    place_record[obj] = {"obj_name": obj_name, "place_on_id": object_id, "place_on_name": om.objects_static[object_id].category}
    if obj_name in object_place['food']['container'] + object_place['stuff']['container']:
        container_list.append(obj)
    else:
        object_list.append(obj)

print('high place pointer num:', len(high_place_pointer))

if keyword == 'highthing':
    # split as much object to the top as possible, but not exceed half of objects.
    # also ensure at least 1 vase
    high_object_num = min(len(remaining_objects)//2, len(high_place_pointer))
    remaining_high_objects, remaining_objects = remaining_objects[:high_object_num - 1], remaining_objects[high_object_num:]
    remaining_high_objects.append('vase_05')
    remaining_objects += ['vase_05']
    remaining_objects += remaining_containers

if len(all_place_pointer) < len(remaining_objects):
    print("WARNING: Not all objects placed, still remaining:")
    print(remaining_objects)
    breakpoint()

np.random.shuffle(all_place_pointer)

#put all object in remaining_objects to all_place_pointer. Note that the number of objects <= the number of place_pointer

for i in range(len(remaining_objects)):
    obj_name = remaining_objects[i]
    place(obj_name, all_place_pointer[i])

if keyword == 'highthing':
    for i in range(len(remaining_high_objects)):
        obj_name = remaining_high_objects[i]
        place(obj_name, high_place_pointer[i])

# put container
if keyword == 'highcontainer':
    if (len(remaining_containers) > len(high_place_pointer)):
        # regenerate remaining containers
        remaining_containers = []
        # first randomly choose object_place['food']['container'] or object_place['stuff']['container']. If for example food is chosen, then see high_place_pointer. 
        # If <= 3, then randomly choose high_place_pointer num container from food container. Else, choose all food container, then choose high_place_pointer - 3 stuff container.
        # Then shuffle
        container_type, opposite_type = random.sample(['food', 'stuff'], 2)
        if len(high_place_pointer) <= 3:
            remaining_containers = random.sample(object_place[container_type]['container'], len(high_place_pointer))
        else:
            remaining_containers = object_place[container_type]['container'] + random.sample(object_place[opposite_type]['container'], len(high_place_pointer) - 3)
        np.random.shuffle(remaining_containers)

    for i in range(len(remaining_containers)):
        obj_name = remaining_containers[i]
        place(obj_name, high_place_pointer[i])

object_probability = 1
curr_occ_map = occ.occupancy_map.copy()

if keyword in ['lowcontainer', 'lowthing']:
    for _ in range(occ.occupancy_map.shape[0] * occ.occupancy_map.shape[1] * 3):
            x_index = np.random.randint(0, occ.occupancy_map.shape[0])
            z_index = np.random.randint(0, occ.occupancy_map.shape[1])
            if check_map(x_index, z_index, curr_occ_map, occ.positions, 0.5): # Unoccupied
                if np.random.random() < object_probability:
                    x = float(occ.positions[x_index, z_index][0])
                    z = float(occ.positions[x_index, z_index][1])
                    room_id = belongs_to_which_room(x, z, scene_bounds)
                    if room_id == -1: continue
                    func_name = get_room_functional_by_id(FLOORPLAN_SCENE_NAME, FLOORPLAN_LAYOUT, room_id)
                    if not check_map(x_index, z_index, curr_occ_map, occ.positions): continue
                    object_room_list.append([obj_name, func_name])
                    obj = c.get_unique_id()
                    if keyword == 'lowcontainer':
                        obj_name = remaining_containers.pop()
                        container_list.append(obj)
                    elif keyword == 'lowthing':
                        obj_name = remaining_low_objects.pop()
                        object_list.append(obj)
                    commands.extend(c.get_add_physics_object(obj_name,
                                                    object_id=obj,
                                                    position={"x": x, "y": 0.1, "z": z},
                                                    rotation={"x": 0, "y": np.random.uniform(0, 360), "z": 0},
                                                    scale_factor = {
                                                    "x": get_scale(obj_name, object_scale),
                                                    "y": get_scale(obj_name, object_scale),
                                                    "z": get_scale(obj_name, object_scale),
                                                    }))
                    curr_occ_map[x_index][z_index] = 1
                    if keyword == 'lowcontainer' and (not remaining_containers): break
                    if keyword == 'lowthing' and (not remaining_low_objects): break
else:
    pass






if keyword == "wheelchair":
    def get_room(pos, x_index, z_index):
        try:
            x = float(pos[x_index, z_index][0])
            z = float(pos[x_index, z_index][1])
            room_id = belongs_to_which_room(x, z, scene_bounds)
        except IndexError:
            room_id = -1
        return room_id
    
    def is_near_wall_and_between_room(occ,pos, x_index, z_index):
        # check four blocks around it. If there is a wall near it (==1), and empty for the rest three block (==0), and between room, return True

        if x_index <= 0 or x_index >= occ.shape[0] - 1 or z_index <= 0 or z_index >= occ.shape[1] - 1:
            return False
        if occ[x_index, z_index] != 0:
            return False
        if occ[x_index - 1, z_index] == 1 and occ[x_index + 1, z_index] == 0 and occ[x_index, z_index - 1] == 0 and occ[x_index, z_index + 1] == 0 and len(set([get_room(pos, x_index, z_index - 1), get_room(pos, x_index, z_index + 1), -1])) == 3:
            return set([get_room(pos, x_index, z_index - 1), get_room(pos, x_index, z_index + 1)])
        if occ[x_index + 1, z_index] == 1 and occ[x_index - 1, z_index] == 0 and occ[x_index, z_index - 1] == 0 and occ[x_index, z_index + 1] == 0 and len(set([get_room(pos, x_index, z_index - 1), get_room(pos, x_index, z_index + 1), -1])) == 3:
            return set([get_room(pos, x_index, z_index - 1), get_room(pos, x_index, z_index + 1)])
        if occ[x_index, z_index - 1] == 1 and occ[x_index + 1, z_index] == 0 and occ[x_index - 1, z_index] == 0 and occ[x_index, z_index + 1] == 0 and len(set([get_room(pos, x_index + 1, z_index), get_room(pos, x_index - 1, z_index), -1])) == 3:
            return set([get_room(pos, x_index + 1, z_index), get_room(pos, x_index - 1, z_index)])
        if occ[x_index, z_index + 1] == 1 and occ[x_index + 1, z_index] == 0 and occ[x_index - 1, z_index] == 0 and occ[x_index, z_index - 1] == 0 and len(set([get_room(pos, x_index + 1, z_index), get_room(pos, x_index - 1, z_index), -1])) == 3:
            return set([get_room(pos, x_index + 1, z_index), get_room(pos, x_index - 1, z_index)])



    def get_shape_z():
        previous_z = None
        shapez = -1

        for x_index in range(occ.occupancy_map.shape[0]):
            for z_index in range(occ.occupancy_map.shape[1]):
                x = float(occ.positions[x_index, z_index][0])
                z = float(occ.positions[x_index, z_index][1])
                shapez += 1
                if previous_z is None:
                    previous_z = z
                if abs(z - previous_z) > 0.01:
                    return shapez

    def pad_and_reshape(occ_map, z):
        # Calculate the current number of elements
        current_elements = occ_map.size

        # Calculate how many elements are needed to be added to make the total number 
        # of elements divisible by z
        rows_needed = -(-current_elements // z)  # This is a ceiling division.
        required_elements = (rows_needed * z) - current_elements

        # Create the new padded array
        padded_array = np.append(occ_map, [-1] * required_elements)

        # Reshape with z columns
        reshaped_array = np.reshape(padded_array, (-1, z))

        return reshaped_array

    shapez = get_shape_z()
    reshaped_occ = pad_and_reshape(occ.occupancy_map, shapez)

    def pad_and_reshape_positions(positions, z):
        # Calculate the current number of elements in the first two dimensions
        current_elements = positions.shape[0] * positions.shape[1]

        # Calculate how many elements are needed to be added to make the total number 
        # of elements divisible by z
        rows_needed = -(-current_elements // z)  # Ceiling division.
        required_elements = (rows_needed * z) - current_elements
        required_rows = required_elements // positions.shape[1]
        
        # Padding for 3D array with shape (-1, positions.shape[1], 2)
        padding = np.full((required_rows, positions.shape[1], 2), [-1, -1])
        
        # Create the new padded array
        padded_array = np.concatenate((positions, padding), axis=0)

        # Reshape with z columns and 2 values in the deepest dimension to represent x and z
        reshaped_array = np.reshape(padded_array, (-1, z, 2))

        return reshaped_array

    reshaped_positions = pad_and_reshape_positions(occ.positions, shapez)
    print(reshaped_positions.shape)
    print(reshaped_occ.shape)

    eazy_read = reshaped_occ.copy()
    for x_index in range(reshaped_occ.shape[0]):
        for z_index in range(reshaped_occ.shape[1]):
            if reshaped_occ[x_index, z_index] == 0:
                eazy_read[x_index, z_index] = get_room(reshaped_positions, x_index, z_index)
            if reshaped_occ[x_index, z_index] == 1:
                eazy_read[x_index, z_index] = 9

    # with open("roomocc.txt", "w") as f:
        # f.write(str(eazy_read))

    # print((x, z)) for each x_index, z_index to a file, round 2 decimal
    # with open("x_z.txt", "w") as f:
    #     for x_index in range(occ.occupancy_map.shape[0]):
    #         for z_index in range(occ.occupancy_map.shape[1]):
    #             x = float(occ.positions[x_index, z_index][0])
    #             z = float(occ.positions[x_index, z_index][1])
    #             f.write(str((round(x, 2), round(z, 2))) + "     ")
    #         f.write("\n")


    def get_adjacent_values(arr, block):
        # This function gathers all unique values adjacent to the -1 block
        adjacent_values = set()
        for x, y in block:
            # Check above
            if x > 0 and arr[x-1, y] != -1:
                adjacent_values.add(arr[x-1, y])
            # Check below
            if x < arr.shape[0] - 1 and arr[x+1, y] != -1:
                adjacent_values.add(arr[x+1, y])
            # Check left
            if y > 0 and arr[x, y-1] != -1:
                adjacent_values.add(arr[x, y-1])
            # Check right
            if y < arr.shape[1] - 1 and arr[x, y+1] != -1:
                adjacent_values.add(arr[x, y+1])
        return adjacent_values

    def dfs(arr, x, y, visited):
        # This is a DFS function to traverse -1 blocks
        if x < 0 or x >= arr.shape[0] or y < 0 or y >= arr.shape[1]:
            return []
        if visited[x, y] or arr[x, y] != -1:
            return []
        
        visited[x, y] = True
        block = [(x, y)]
        
        # Recursively search adjacent cells
        block.extend(dfs(arr, x+1, y, visited))
        block.extend(dfs(arr, x-1, y, visited))
        block.extend(dfs(arr, x, y+1, visited))
        block.extend(dfs(arr, x, y-1, visited))
        
        return block

    def find_blocks_and_calculate_centroid(arr):
        visited = np.full(arr.shape, False)
        blocks_dict = {}

        for x in range(arr.shape[0]):
            for y in range(arr.shape[1]):
                if arr[x, y] == -1 and not visited[x, y]:
                    block = dfs(arr, x, y, visited)
                    adj_values = get_adjacent_values(arr, block)
                    
                    # Ensure the adjacent values contain only two different numbers plus 9
                    if len(adj_values) == 3 and 9 in adj_values:
                        adj_values.remove(9)
                        adj_values_tuple = tuple(sorted(adj_values))
                        centroid_x = sum(point[0] for point in block) / len(block)
                        centroid_y = sum(point[1] for point in block) / len(block)
                        centroid = (centroid_x, centroid_y)
                        
                        if adj_values_tuple not in blocks_dict:
                            blocks_dict[adj_values_tuple] = []
                        blocks_dict[adj_values_tuple].append(centroid)
        
        return blocks_dict



    # # record whether there are obstacle between two rooms
    # obstacle_record = {}
    # roompairpos = {}
    # for x_index in range(reshaped_occ.shape[0]):
    #     for z_index in range(reshaped_occ.shape[1]):
    #         if roompair := is_near_wall_and_between_room(reshaped_occ, reshaped_positions, x_index, z_index):
    #             roompair = tuple(sorted(roompair))
    #             if roompair not in obstacle_record:
    #                 obstacle_record[roompair] = 0
    #             obstacle_record[roompair] += 1
    #             if roompair not in roompairpos:
    #                 roompairpos[roompair] = []
    #             x = float(reshaped_positions[x_index, z_index][0])
    #             z = float(reshaped_positions[x_index, z_index][1])
    #             roompairpos[roompair].append(np.array([x,z]))
    # print(f'{obstacle_record=}')

    roompairpos = find_blocks_and_calculate_centroid(eazy_read)
    print('roompairpos', roompairpos)

    for k,v in roompairpos.items():
            v = v[0]
            x = float(reshaped_positions[int(v[0]), int(v[1])][0])
            z = float(reshaped_positions[int(v[0]), int(v[1])][1])
            obj_name = random.choice(object_place['obstacle'])
            commands.extend(c.get_add_physics_object(obj_name,
                                                object_id=c.get_unique_id(),
                                                position={"x": x, "y": 0.1, "z": z},
                                                rotation={"x": 0, "y": np.random.uniform(0, 360), "z": 0},
                                                scale_factor = {
                                                "x": get_scale(obj_name, object_scale),
                                                "y": get_scale(obj_name, object_scale),
                                                "z": get_scale(obj_name, object_scale),
                                            }))



# commands.append({"$type": "step_physics", "frames": 100})



# Save commands and metadata to json


response = c.communicate(commands)


om = ObjectManager(transforms=True, rigidbodies=True, bounds=True)

c.communicate([])
c.communicate([])
c.add_ons.extend([om])
c.communicate([])
c.communicate([])

for object_id in place_record.keys():
    place_record[object_id]["initial_height"] = float(om.bounds[object_id].bottom[1])
c.communicate([{"$type": "step_physics", "frames": 100}])
c.communicate([])


for trynum in range(100):
    task_is_ok = True
    for object_id in place_record.keys():
        if place_record[object_id]['initial_height'] - float(om.bounds[object_id].bottom[1]) > 0.4 and float(om.bounds[object_id].bottom[1]) < 0.1:
            # print(f"WARNING: Object {object_id} may fall down: info {place_record[object_id]}, final height {float(om.bounds[object_id].bottom[1])}")
            # if task-related, exit
            error1 = place_record[object_id]['obj_name'] in [i[0] for i in task]
            error2 = place_record[object_id]['obj_name'] in object_place['food']['container'] and any([i[0] in object_place['food']['target'] for i in task])
            error3 = place_record[object_id]['obj_name'] in object_place['stuff']['container'] and any([i[0] in object_place['stuff']['target'] for i in task])
            if error1 or error2 or error3:
                task_is_ok = False
                # print("normal falldown error, regenerate task")
                task = generate_task()
                break
    if task_is_ok is True and keyword == "highthing":
        # make sure at least 2 task objects are placed on high place
        assert remaining_high_objects
        # count how many remaining_high_objects in [i[0] for i in task]
        count = 0
        for i in task:
            if i[0] in remaining_high_objects:
                count += 1
        if count < 2:
            task_is_ok = False
            # print("False because highthing, regenerate task")
            # print(f"{task=},\n{remaining_high_objects=}")
            # breakpoint()
            task = generate_task()


    if task_is_ok:
        break
    if trynum == 99:
        print("Cannot find good task, exit.")
        exit(-1)

print("task ", task)
print()

metadata = {
    'floorplan_scene_name': FLOORPLAN_SCENE_NAME,
    'floorplan_layout': FLOORPLAN_LAYOUT,
    'scene_id': scene_id,
    'object_room_list': object_room_list,
    'task':{
    'goal_position_names': ['bed' if keyword != 'highgoalplace' else random.choice(high_position_name)],
    'goal_task': task,
    'container_names': list(object_place['food']['container'] + object_place['stuff']['container']),
    'task_kind': keyword,
    'constraint_type': constraint_type,
    }
}
if keyword == "wheelchair":
    metadata['task']["obstacle_names"] = object_place['obstacle']

metadata['object_height'] = [round(float(om.bounds[object_id].bottom[1]) ,2) for object_id in object_list]
metadata['container_height'] = [round(float(om.bounds[object_id].bottom[1]) ,2) for object_id in container_list]

print(metadata)
print()
print(f"{len(all_place_pointer)=}, {len(remaining_objects)=}")


with open(f"./dataset/{dataset_prefix}/{FLOORPLAN_SCENE_NAME}_{FLOORPLAN_LAYOUT}_{scene_id}.json", "w") as f:
    f.write(json.dumps(commands, indent=4))
with open(f"./dataset/{dataset_prefix}/{FLOORPLAN_SCENE_NAME}_{FLOORPLAN_LAYOUT}_{scene_id}_metadata.json", "w") as f:
    f.write(json.dumps(metadata, indent=4))

c.communicate({"$type": "terminate"})
