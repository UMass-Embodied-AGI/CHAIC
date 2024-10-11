import numpy as np
from tdw.replicant.action_status import ActionStatus
from transport_challenge_multi_agent.transport_challenge import TransportChallenge
from transport_challenge_multi_agent.outdoor_transport_challenge import OutdoorTransportChallenge
from tdw.replicant.image_frequency import ImageFrequency
from tdw.add_ons.image_capture import ImageCapture
from tdw.output_data import OutputData, SegmentationColors, CameraMatrices
import random
import time
import json
import os
import argparse


def generate_imgs(save_dir, data_prefix, scene, layout, layout2, task, meta_data):
    _start = time.time()
    print("Start ---------------------------------", time.strftime("%H:%M:%S", time.localtime(_start)))

    c = TransportChallenge(
        check_version=True, screen_width=512, screen_height=512,
        image_frequency=ImageFrequency.always, png=False, image_passes=["_img", "_id", "_depth"],
        launch_build=True, port = 1999)
    print("Build Scene ---------------------------------", time.time() - _start)
    c.start_floorplan_trial(scene=scene, layout=str(layout) + "_" + str(layout2), replicants=1,
                            random_seed=None, data_prefix=data_prefix)  # Random Spawn Replicant
    if task == "outdoor_shopping":
        rand_x = random.uniform(metadata["border"]["x_min"], metadata["border"]["x_max"])
        rand_z = random.uniform(metadata["border"]["z_min"], metadata["border"]["z_max"])
        pos = {"x": rand_x, "y": 0, "z": rand_z}
        c.communicate({"$type": "teleport_object", "position": pos, "id": 0})

    print("Floor -------------------------------", time.time() - _start)
    c.communicate([{"$type": "set_floorplan_roof", "show": False}])
    print("Sky -------------------------------", time.time() - _start)

    c.communicate({"$type": "add_hdri_skybox", "name": "sky_white",
                                 "url": "https://tdw-public.s3.amazonaws.com/hdri_skyboxes/linux/2019.1/sky_white",
                                 "exposure": 2, "initial_skybox_rotation": 0, "sun_elevation": 90,
                                 "sun_initial_angle": 0, "sun_intensity": 1.25})
    print("FOV -------------------------------", time.time() - _start)
    c.communicate({"$type": "set_field_of_view", "avatar_id" : str(0), "field_of_view" : 90})

    print("Setting -------------------------------", time.time() - _start)
    capture = ImageCapture(avatar_ids=[c.replicants[0].static.avatar_id], path=save_dir,
                           pass_masks=["_img", "_depth", "_id"], png=False)
    c.add_ons.extend([capture])
    c.replicants[0].collision_detection.avoid = False
    obj_ids = save_seg_color_dict(c, data_prefix, save_dir)
    print(time.time() - _start)

    print("Move ----------------------------------", time.time() - _start)
    random_sample(c, obj_ids, save_dir, metadata)
    c.communicate({"$type": "terminate"})

def check_border(border, pos):
    if border['x_min'] < pos['x'] and pos['x'] < border['x_max'] and border['z_min'] < pos['z'] and pos['z'] < border['z_max']:
        return True
    else:
        return False

def random_sample(c, obj_ids, save_dir, metadata):
    rooms = list(c._get_rooms_map(communicate=True).values())
    x_min = np.min([np.min([pos['x'] for pos in room]) for room in rooms])
    x_max = np.max([np.max([pos['x'] for pos in room]) for room in rooms])
    random.shuffle(rooms)
    replicant = c.replicants[0]
    pre_pos = None
    with open(os.path.join(save_dir, "logger.txt"), "w") as f:
        for k, region in enumerate(rooms):
            f.write(f"Room: {k}\n")
            step_count = 0
            try_count = 0
            while (step_count <= 500) and (try_count <= 1000):
                pos = random.choice(region)
                while "border" in metadata and not check_border(metadata["border"], pos):
                    # TODO: Border is used in outside scene
                    # In this scene, the region given by environment is wrong
                    # Many free space is mistakenly marked as occupied by TDW
                    # So we directly sample in the border
                    rand_x = random.uniform(metadata["border"]["x_min"], metadata["border"]["x_max"])
                    rand_z = random.uniform(metadata["border"]["z_min"], metadata["border"]["z_max"])
                    pos = {"x": rand_x, "y": 0, "z": rand_z}
                replicant.navigate_to(target=pos)
                print(pos)
                while replicant.action.status in [ActionStatus.ongoing]:
                    print("status:", replicant.action.status)
                    c.communicate([])
                    try_count += 1
                    cur_pos = replicant.dynamic.transform.position
                    if not np.all(pre_pos == cur_pos):
                        step_count += 1
                        pre_pos = cur_pos

                c.communicate([])
                print("status:", replicant.action.status, "move_step:", step_count, "commend_step", try_count)
                f.write(f"{step_count}: {replicant.action.status}\n")

def save_seg_color_dict(c : TransportChallenge, data_prefix, save_dir):
    with open(os.path.join(data_prefix, "name_map.json")) as json_file:
        perception_obj_list = json.load(json_file).keys()

    color_obj_map = dict() # color -> object idx
    obj_ids = []
    response = c.communicate({"$type": "send_segmentation_colors", "frequency": "once"})
    byte_array = filter(lambda x: OutputData.get_data_type_id(x) == "segm", response).__next__()
    seg = SegmentationColors(byte_array)
    for i in range(seg.get_num()):
        _obj = seg.get_object_name(i).lower()
        _color = seg.get_object_color(i)
        _id = seg.get_object_id(i)

        if _obj in perception_obj_list:
            color_obj_map[str(_color)] = _obj
            obj_ids.append(_id)
        # goal place is specially defined
        elif seg.get_object_category(i).lower() in ["bed", "cabinet", "refrigerator", "fire hydrant", "truck"]:
            assert seg.get_object_name(i).lower() not in ["bed", "cabinet", "refrigerator", "fire hydrant", "truck"], "Object name is duplicated"
            if 'door' in seg.get_object_name(i).lower():
                continue
            color_obj_map[str(_color)] = seg.get_object_category(i).lower()
            obj_ids.append(_id)

    with open(os.path.join(save_dir, "color_obj_map.json"), "w") as f:
        f.write(json.dumps(color_obj_map))
    return obj_ids


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir", type=str, default="captured_imgs_helper")
    parser.add_argument("--train_data_prefix", type=str, default="dataset/train_dataset")
    parser.add_argument("--test_data_prefix", type=str, default="dataset/test_dataset")
    parser.add_argument("--tasks", nargs='+', default=("wheelchair", "lowthing", "normal", "highcontainer", "highthing", "highgoalplace"), type=str, help="which task to collect data")
    # All tasks are ("wheelchair", "lowthing", "normal", "highcontainer", "highthing", "highgoalplace", "outdoor_with_bike")
    parser.add_argument("--train_scene", nargs='+', default=("1a", "4a"), type=str, help="which scenes to train data")
    parser.add_argument("--test_scene", nargs='+', default=("2a", ), type=str, help="which scenes to test data")
    parser.add_argument("--train_layout", nargs='+', default=(0, 1, 2), type=int, help="which layout to train data")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    train_save_dir = os.path.join(args.save_dir, 'train')
    for j, task in enumerate(args.tasks):
        train_dataset = os.path.join(args.train_data_prefix, task)
        layout_type = [0]
        if task in ['wheelchair', 'highthing']:
            # There are special object in it, so we collect all the data here.
            layout_type = args.train_layout
        if task in ['outdoor_shopping', 'outdoor_furniture']:
            scene_list = ['2a']
            layout_type = [0]
            number_list = range(12)
        else:
            scene_list = args.train_scene
            number_list = range(1)
        for i in number_list:
            print(i)
            for layout in layout_type:
                for scene in scene_list:
                    matadata = os.path.join(train_dataset, f"{scene}_{layout}_{i}_metadata.json")
                    print(matadata)
                    assert os.path.exists(matadata)
                    with open(matadata) as f:
                        metadata = json.load(f)
                    if task in ['outdoor_shopping']:
                        metadata["border"] = {
                            "x_min": 5,
                            "x_max": 25,
                            "z_min": -2,
                            "z_max": 3.5
                        }
                    elif task in ['outdoor_furniture']:
                        metadata["border"] = {
                            "x_min": -3,
                            "x_max": 10,
                            "z_min": 0,
                            "z_max": 5
                        }
                    _dir = os.path.join(train_save_dir, f"{scene}_{layout}_{task}")
                    if os.path.exists(_dir):
                        continue
                    print(_dir)
                    generate_imgs(_dir, train_dataset, scene, layout, i, task, metadata)

    test_save_dir = os.path.join(args.save_dir, 'test')
#    tasks = ['normal_debug', 'highcontainer_debug', 'highthing_debug', 'lowcontainer_debug', 'lowthing_debug', 'obstacles_debug']
#    tasks = []
    for j, task in enumerate(args.tasks):
        layout_type = [0]
        test_dataset = os.path.join(args.test_data_prefix, task)
        if task == "outdoor_shopping":
            num = 2
        elif task == "outdoor_furniture":
            num = 12
        else:
            num = 1
        for i in range(num):
            for layout in layout_type:
                for scene in args.test_scene:
                    matadata = os.path.join(test_dataset, f"{scene}_{layout}_{i}_metadata.json")
                    print(matadata)
                    assert os.path.exists(matadata)
                    with open(matadata) as f:
                        metadata = json.load(f)
                    if task in ['outdoor_shopping']:
                        metadata["border"] = {
                            "x_min": 5,
                            "x_max": 25,
                            "z_min": -2,
                            "z_max": 3.5
                        }
                    elif task in ['outdoor_furniture']:
                        metadata["border"] = {
                            "x_min": -3,
                            "x_max": 10,
                            "z_min": 0,
                            "z_max": 5
                        }
                    _dir = os.path.join(test_save_dir, f"{scene}_{layout}_{i + j * 2}_{task}")
                    if os.path.exists(_dir):
                        continue
                    print(_dir)
                    generate_imgs(_dir, test_dataset, scene, layout, i, task, metadata)