from csv import DictReader
from typing import List, Dict, Union, Tuple, Optional
from subprocess import run, PIPE
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from tdw.version import __version__
from tdw.controller import Controller
from tdw.tdw_utils import TDWUtils
from tdw.add_ons.floorplan import Floorplan
from tdw.add_ons.replicant import Replicant
from tdw.librarian import HumanoidLibrarian
from tdw.add_ons.occupancy_map import OccupancyMap
from tdw.add_ons.interior_scene_lighting import InteriorSceneLighting
from tdw.add_ons.object_manager import ObjectManager
from tdw.add_ons.nav_mesh import NavMesh
from tdw.add_ons.container_manager import ContainerManager
from tdw.replicant.image_frequency import ImageFrequency
from tdw.replicant.action_status import ActionStatus
from tdw.scene_data.scene_bounds import SceneBounds
from transport_challenge_multi_agent.challenge_state import ChallengeState
from transport_challenge_multi_agent.replicant_transport_challenge import ReplicantTransportChallenge
from transport_challenge_multi_agent.wheelchair_replicant_transport_challenge import ReplicantTransportChallenge as WheelchairReplicantTransportChallenge
from transport_challenge_multi_agent.paths import CONTAINERS_PATH, TARGET_OBJECTS_PATH, TARGET_OBJECT_MATERIALS_PATH
from transport_challenge_multi_agent.globals import Globals
from transport_challenge_multi_agent.asset_cached_controller import AssetCachedController
from transport_challenge_multi_agent.agent_ability_info import ability_mapping
from tdw.add_ons.logger import Logger
import os
import json


class TransportChallenge(AssetCachedController):
    """
    A subclass of `Controller` for the Transport Challenge. Always use this class instead of `Controller`.

    See the README for information regarding output data and scene initialization.
    """

    """:class_var
    The mass of each target object.
    """
    TARGET_OBJECT_MASS: float = 0.25
    """:class_var
    If an object is has this mass or greater, the object will become kinematic.
    """
    KINEMATIC_MASS: float = 100
    """:class_var
    The expected version of TDW.
    """
    TDW_VERSION: str = "1.12.27.0"
    """:class_var
    The goal zone is a circle defined by `self.goal_center` and this radius value.
    """
    GOAL_ZONE_RADIUS: float = 1

    def __init__(self, port: int = 1071, check_version: bool = True, launch_build: bool = True, screen_width: int = 256,
                 screen_height: int = 256, image_frequency: ImageFrequency = ImageFrequency.once, png: bool = True, asset_cache_dir = "transport_challenge_asset_bundles",
                 image_passes: List[str] = None, target_framerate: int = 250, enable_collision_detection: bool = False, logger_dir = None, replicants_name=['replicant_0', 'replicant_0'], replicants_ability = ['helper', 'helper']):
        """
        :param port: The socket port used for communicating with the build.
        :param check_version: If True, the controller will check the version of the build and print the result.
        :param launch_build: If True, automatically launch the build. If one doesn't exist, download and extract the correct version. Set this to False to use your own build, or (if you are a backend developer) to use Unity Editor.
        :param screen_width: The width of the screen in pixels.
        :param screen_height: The height of the screen in pixels.
        :param image_frequency: How often each Replicant will capture an image. `ImageFrequency.once` means once per action, at the end of the action. `ImageFrequency.always` means every communicate() call. `ImageFrequency.never` means never.
        :param png: If True, the image pass from each Replicant will be a lossless png. If False, the image pass from each Replicant will be a lossy jpg.
        :param image_passes: A list of image passes, such as `["_img"]`. If None, defaults to `["_img", "_id", "_depth"]` (i.e. image, segmentation colors, depth maps).
        :param target_framerate: The target framerate. It's possible to set a higher target framerate, but doing so can lead to a loss of precision in agent movement.
        """

        try:
            q = run(["git", "rev-parse", "--show-toplevel"], stdout=PIPE)
            p = Path(str(q.stdout.decode("utf-8").strip())).resolve()
            if "CHAIC" not in p.stem:
                print("Warning! You might be using code copied from the Co-LLM-Agents repo. Your code might be out of date.\n")
        except OSError:
            pass
        if TransportChallenge.TDW_VERSION != __version__:
            print(f"Warning! Your local install of TDW is version {__version__} but the Multi-Agent Transport Challenge requires version {TransportChallenge.TDW_VERSION}\n")
        super().__init__(cache_dir=asset_cache_dir, port=port, check_version=check_version, launch_build=launch_build)
        if logger_dir is not None:
            self.logger = Logger(path=os.path.join(logger_dir, "action_log.log"))
            self.add_ons.append(self.logger)
        else:
            self.logger = None
        self._image_frequency: ImageFrequency = image_frequency
        """:field
        A dictionary of all Replicants in the scene. Key = The Replicant ID. Value = [`ReplicantTransportChallenge`](replicant_transport_challenge.md).
        """
        self.replicants: Dict[int, ReplicantTransportChallenge] = dict()
        self.replicants_name = replicants_name
        self.replicants_ability_name = replicants_ability
        self.replicants_ability = [ability_mapping[ability]() for ability in replicants_ability]
        """:field
        The `ChallengeState`, which includes container IDs, target object IDs, containment state, and which Replicant is holding which objects.
        """
        self.state: ChallengeState = ChallengeState()
        """:field
        An [`OccupancyMap`](https://github.com/threedworld-mit/tdw/blob/master/Documentation/lessons/navigation/occupancy_maps.md). This is used to place objects and can also be useful for navigation.
        """
        self.occupancy_map: OccupancyMap = OccupancyMap()
        """:field
        An [`ObjectManager`](https://github.com/threedworld-mit/tdw/blob/master/Documentation/lessons/core_concepts/output_data.md#the-objectmanager-add-on) add-on. This is useful for referencing all objects in the scene.
        """
        self.object_manager: ObjectManager = ObjectManager(transforms=True, rigidbodies=False, bounds=True)
        """:field
         The challenge is successful when the Replicants move all of the target objects to the the goal zone, which is defined by this position and `TransportChallenge.GOAL_ZONE_RADIUS`. This value is set at the start of a trial.
        """
        self.goal_position: np.ndarray = np.zeros(shape=3)
        # Initialize the random state. This will be reset later.
        self._rng: np.random.RandomState = np.random.RandomState()
        # All possible target objects. Key = name. Value = scale.
        self._target_objects_names_and_scales: Dict[str, float] = dict()
        self.scene_bounds = None
        with open(str(TARGET_OBJECTS_PATH.resolve())) as csvfile:
            reader = DictReader(csvfile)
            for row in reader:
                self._target_objects_names_and_scales[row["name"]] = float(row["scale"])
        self._target_object_names: List[str] = list(self._target_objects_names_and_scales.keys())
        self._target_object_visual_materials: List[str] = TARGET_OBJECT_MATERIALS_PATH.read_text().split("\n")
        # Get all possible container names.
        self._container_names: List[str] = CONTAINERS_PATH.read_text().split("\n")
        self._scene_bounds: Optional[SceneBounds] = None
        self.object_property = dict()
        if image_passes is None:
            self._image_passes: List[str] = ["_img", "_id", "_depth"]
        else:
            self._image_passes: List[str] = image_passes
        self._target_framerate: int = target_framerate
        # Initialize the window and rendering.
        self.communicate([{"$type": "set_screen_size",
                           "width": screen_width,
                           "height": screen_height},
                          {"$type": "set_img_pass_encoding",
                           "value": png},
                          {"$type": "set_render_quality",
                           "render_quality": 5}])
        # Sett whether we need to scale IK motion duration.
        Globals.SCALE_IK_DURATION = self._is_standalone
        self.enable_collision_detection = enable_collision_detection
        self.rooms_name = None
        # # Use Humanoid librian as default
        # self.HUMANOID_LIBRARIANS[Replicant.LIBRARY_NAME] = HumanoidLibrarian("humanoids.json")
    def start_floorplan_trial(self, scene: str, layout: int, task_meta = None,
                              replicants: Union[int, List[Union[int, np.ndarray, Dict[str, float]]]] = 2,
                              lighting: bool = True, random_seed: int = None, data_prefix = 'dataset/dataset_train', object_property: dict = dict()) -> None:
        """
        Start a trial in a floorplan scene.

        :param scene: The scene. Options: "1a", "1b", "1c", "2a", "2b", "2c", "4a", "4b", "4c", "5a", "5b", "5c"
        :param layout: The layout of objects in the scene. Options: 0, 1, 2.
        :param num_containers: The number of containers in the scene.
        :param num_target_objects: The number of target objects in the scene.
        :param container_room_index: The index of the room in which containers will be placed. If None, the room is random.
        :param target_objects_room_index: The index of the room in which target objects will be placed. If None, the room is random.
        :param goal_room_index: The index of the goal room. If None, the room is random.
        :param replicants: An integer or a list. If an integer, this is the number of Replicants in the scene; they will be added at random positions on the occupancy map. If a list, each element can be: An integer (the Replicant will be added *in this room* at a random occupancy map position), a numpy array (a worldspace position), or a dictionary (a worldspace position, e.g. `{"x": 0, "y": 0, "z": 0}`.
        :param lighting: If True, add an HDRI skybox for realistic lighting. The skybox will be random, using the `random_seed` value below.
        :param random_seed: The random see used to add containers, target objects, and Replicants, as well as to set the lighting and target object materials. If None, the seed is random.
        """
        floorplan = Floorplan()
        if type(layout) == int:
            floorplan.init_scene(scene=scene, layout=layout) # 0 / 1 / 2
        else:
            floorplan.init_scene(scene=scene, layout=int(layout[0])) # 0_1: the first pos is the layout, the second pos is the container setting
        self.add_ons.append(floorplan)
        if lighting:
            self.add_ons.append(InteriorSceneLighting(rng=np.random.RandomState(random_seed)))
        self.communicate([])
        self.scene = scene
        self.layout = layout
        self.data_prefix = data_prefix
        self.object_property = object_property
        self._start_trial(replicants=replicants, task_meta = task_meta, random_seed=random_seed)

    def communicate(self, commands: Union[dict, List[dict]]) -> list:
        """
        Send commands and receive output data in response.

        :param commands: A list of JSON commands.

        :return The output data from the build.
        """
        #print(commands)
        return super().communicate(commands)

    def _start_trial(self, replicants: Union[int, List[Union[int, np.ndarray, Dict[str, float]]]] = 2, task_meta = None, random_seed: int = None) -> None:
        """
        Start a trial in a floorplan scene.
        food or stuff
        """
        self.communicate({"$type": "set_floorplan_roof", "show": False})
        load_path = os.path.join(self.data_prefix, f"{self.scene}_{self.layout}.json")
        with open(load_path, "r") as f: scene = json.load(f)
        if os.path.exists(os.path.join(self.data_prefix, f"{self.scene}_{self.layout}_metadata.json")):
            load_count_and_position_path = os.path.join(self.data_prefix, f"{self.scene}_{self.layout}_metadata.json")
        with open(load_count_and_position_path, "r") as f: count_and_position = json.load(f)

        current_obj = None
        scene_with_filter = []
        for commands in scene:
            if commands['$type'] == 'add_object':
                current_obj = commands['name']
            if current_obj is not None and current_obj in self.object_property.keys():
                if "container" in commands['$type'] and self.object_property[current_obj] != "container":
                    # remove this commands
                    print(f"Remove {commands} since it is not a container.")
                    continue
            scene_with_filter.append(commands)

        self.communicate(scene_with_filter)
        self.state = ChallengeState()
        self.add_ons.clear() # Clear the add-ons.
        if self.logger is not None:
            self.add_ons.append(self.logger)
        self.replicants.clear()
        # Add an occupancy map.
        self.add_ons.append(self.occupancy_map)
        # Get the rooms.
        rooms: Dict[int, List[Dict[str, float]]] = self._get_rooms_map(communicate=True)
        replicant_positions: List[Dict[str, float]] = list()

        # Spawn a certain number of Replicants in a certain rooms with random position
        if isinstance(replicants, int):
            # Randomize the rooms.
            room_indices = [x for x in rooms.keys() if len(rooms[x]) >= 9]
            room_index = self._rng.randint(0, len(room_indices))
            # Place Replicants in different rooms.
            for i in range(replicants):
                # Get a random position in the room.
                if self.layout != 0:
                    room_index = self._rng.randint(0, len(room_indices))
                positions = rooms[room_indices[room_index]]
                replicant_positions.append(positions[self._rng.randint(0, len(positions))])
                rooms = self._occupy_position(position=replicant_positions[-1])

        # Add the Replicants. If the position is fixed, the position is the same as the last time.
        # Only if the agent is not in the count_and_position, we will use the random position and save it.
        save_tag = False
        if "agent" in count_and_position:
           for i in range(replicants):
                if str(i) in count_and_position["agent"]:
                    replicant_positions[i] = count_and_position["agent"][str(i)]
                else:
                    count_and_position["agent"][str(i)] = replicant_positions[i]
                    save_tag = True
        else:
            save_tag = True
        count_and_position["agent"] = {str(i): replicant_positions[i] for i in range(replicants)}
        for i, replicant_position in enumerate(replicant_positions):
            if self.replicants_ability_name[i] == 'wheelchair':
                #TODO: make it into use
                replicant = ReplicantTransportChallenge(replicant_id=i,
                                                        state=self.state,
                                                        position=replicant_position,
                                                        image_frequency=self._image_frequency,
                                                        target_framerate=self._target_framerate,
                                                        enable_collision_detection=self.enable_collision_detection,
                                                        name=self.replicants_name[i],
                                                        ability=self.replicants_ability[i]
                                                        )
            else:
                replicant = ReplicantTransportChallenge(replicant_id=i,
                                                        state=self.state,
                                                        position=replicant_position,
                                                        image_frequency=self._image_frequency,
                                                        target_framerate=self._target_framerate,
                                                        enable_collision_detection=self.enable_collision_detection,
                                                        name=self.replicants_name[i],
                                                        ability=self.replicants_ability[i]
                                                        )
            self.replicants[replicant.replicant_id] = replicant
            self.add_ons.append(replicant)
            
        # Set the pass masks.
        # Add a challenge state and object manager.
        self.object_manager.reset()
        self.add_ons.extend([self.state, self.object_manager])
        self.communicate([])
        commands = []
        # since the fridge is easy to drop, I want to make it kinematic.
        for object_id in self.object_manager.objects_static:
            o = self.object_manager.objects_static[object_id]
            if o.category == "refrigerator" or o.category == "cabinet":
                commands.append({"$type": "set_kinematic_state",
                                 "id": object_id,
                                 "is_kinematic": True,
                                 "use_gravity": False})
                self.object_manager.objects_static[object_id].kinematic = True

        if task_meta is not None:
            if 'goal_object_names' not in task_meta.keys():
                # For indoor scenes: not goal object name, but possible tasks
                task_meta['goal_object_names'] = []
                # task_meta['possible_object_names'] = []
                for it in task_meta['goal_task']:
                    # name & number
                    task_meta['goal_object_names'].append(it[0])
                # for task in task_meta['possible_task']:
                #     for it in task:
                #         if it[0] not in task_meta['possible_object_names']:
                #             task_meta['possible_object_names'].append(it[0])

            for object_id in self.object_manager.objects_static.keys():
                if self.object_manager.objects_static[object_id].name in task_meta['goal_object_names']:
                    self.state.target_object_ids.append(object_id)
                if self.object_manager.objects_static[object_id].name in task_meta['possible_object_names']:
                    self.state.possible_target_object_ids.append(object_id)
                if self.object_manager.objects_static[object_id].name in task_meta['container_names']:
                    self.state.container_ids.append(object_id)
                if self.object_manager.objects_static[object_id].name in 'vase':
                    self.state.vase_ids.append(object_id)
                if 'obstacle_names' in task_meta.keys():
                    if self.object_manager.objects_static[object_id].name in task_meta['obstacle_names']:
                        self.state.obstacle_ids.append(object_id)
        for replicant_id in self.replicants:
            # Set pass masks.
            commands.append({"$type": "set_pass_masks",
                             "pass_masks": self._image_passes,
                             "avatar_id": self.replicants[replicant_id].static.avatar_id})
            # Ignore collisions with target objects.
            self.replicants[replicant_id].collision_detection.exclude_objects.extend(self.state.target_object_ids)
        # Add a NavMesh.
        nav_mesh_exclude_objects = list(self.replicants.keys())
        nav_mesh_exclude_objects.extend(self.state.target_object_ids)
        nav_mesh = NavMesh(exclude_objects=nav_mesh_exclude_objects)
        self.add_ons.append(nav_mesh)
        self.container_manager = ContainerManager()
        self.add_ons.append(self.container_manager)
        # Send the commands.
        # self.communicate(commands)
        # Reset the heads.
        for replicant_id in self.replicants:
            self.replicants[replicant_id].reset_head(scale_duration=Globals.SCALE_IK_DURATION)
        reset_heads = False
        while not reset_heads:
            self.communicate([])
            reset_heads = True
            for replicant_id in self.replicants:
                if self.replicants[replicant_id].action.status == ActionStatus.ongoing:
                    reset_heads = False
                    break
        resp = self.communicate([{"$type": "send_scene_regions"}])
        self.scene_bounds = SceneBounds(resp=resp)
        for replicant_id in self.replicants:
            self.replicants[replicant_id].scene_bounds = self.scene_bounds

        # Bright lights.
        self.communicate({"$type": "add_hdri_skybox", \
                        "name": "sky_white", \
                        "url": "https://tdw-public.s3.amazonaws.com/hdri_skyboxes/linux/2019.1/sky_white", \
                        "exposure": 2, "initial_skybox_rotation": 0, "sun_elevation": 90, \
                        "sun_initial_angle": 0, "sun_intensity": 1.25})

        # Save the position of agents.
        if save_tag:
             with open(load_count_and_position_path, "w") as f:
                 json.dump(count_and_position, f, indent=4)
             print(load_count_and_position_path, 'saved')

    def _get_rooms_map(self, communicate: bool) -> Dict[int, List[Dict[str, float]]]:
        # Generate a new occupancy map and request scene regions data.
        if communicate:
            self.occupancy_map.generate()
            resp = self.communicate([{"$type": "send_scene_regions"}])
            self._scene_bounds = SceneBounds(resp=resp)
        rooms: Dict[int, List[Dict[str, float]]] = dict()
        for ix in range(self.occupancy_map.occupancy_map.shape[0]):
            for iz in range(self.occupancy_map.occupancy_map.shape[1]):
                # Ignore non-free positions.
                if self.occupancy_map.occupancy_map[ix][iz] != 0:
                    continue
                # Find the room that this position is in.
                p = self.occupancy_map.positions[ix][iz]
                for i, region in enumerate(self._scene_bounds.regions):
                    if region.is_inside(p[0], p[1]) and region.is_inside(p[0] - 0.75, p[1] - 0.75) and region.is_inside(p[0] + 0.75, p[1] + 0.75):
                        # Make sure it is not near a wall.
                        if i not in rooms:
                            rooms[i] = list()
                        rooms[i].append({"x": float(p[0]),
                                         "y": 0,
                                         "z": float(p[1])})
                        break
        return rooms

    def _occupy_position(self, position: Dict[str, float]) -> Dict[int, List[Dict[str, float]]]:
        origin = TDWUtils.vector3_to_array(position)
        for ix in range(self.occupancy_map.occupancy_map.shape[0]):
            for iz in range(self.occupancy_map.occupancy_map.shape[1]):
                # Ignore non-free positions.
                if self.occupancy_map.occupancy_map[ix][iz] != 0:
                    continue
                # Find the room that this position is in.
                p2 = self.occupancy_map.positions[ix][iz]
                p3 = np.array([p2[0], 0, p2[1]])
                if np.linalg.norm(origin - p3) <= 0.5:
                    self.occupancy_map.occupancy_map[ix][iz] = 1
        return self._get_rooms_map(communicate=False)