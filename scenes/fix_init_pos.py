import os
import json
from transport_challenge_multi_agent.transport_challenge import TransportChallenge

scenes = ['normal', 'highcontainer', 'highthing', 'lowthing', 'highgoalplace', 'wheelchair']
data_root = 'dataset'

for scene in scenes:
    for task in ['train_dataset', 'test_dataset']:
        scene_path = os.path.join(data_root, task, scene)
        for root, dirs, files in os.walk(scene_path):
            for file in files:
                if file.endswith('metadata.json'):
                    print(f"Found: {scene_path}/{file}")
                    with open(os.path.join(scene_path, file), 'r') as f:
                        metadata = json.load(f)
                        if "agent" in metadata:
                            continue

                    controller = TransportChallenge(port=11111)
                    controller.start_floorplan_trial(
                        scene = file[:2],
                        layout = file[3: 6],
                        data_prefix = scene_path,
                        replicants = 2
                    )
                    controller.communicate({"$type": "terminate"})
                    controller.socket.close()