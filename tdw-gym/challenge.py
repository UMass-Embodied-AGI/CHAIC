import argparse
import os
import json
import gym
import time
import pickle
import logging
import sys
import shutil
import numpy as np
from typing import List
from enum import Enum

# add this dictionary to python env path:
base_path = os.getcwd()
sys.path.append(base_path)

class PlanStatus(Enum):
    goto_target = 0
    goto_container = 1
    goto_drop_zone = 2
    # do not change 0, 1, 2
    put_in = 3
    explore = 4
    wait = 5
    remove_obstacle = 6

from tdw_gym import TDW
from plan_agent import PlanAgent
from plan_agent_for_bike_agent import PlanAgentForBikeAgent
from plan_agent_for_furniture_agent import PlanAgentForFurnitureAgent
from child_agent import ChildAgent
from red_light_agent import RedLightAgent
from follow_agent import FollowAgent
# from lm_agent import lm_agent
# from lm_helper_agent import lm_helper_agent
# from random_agent import random_agent
import transport_challenge_multi_agent.utils as utils
# try:
#     gym.envs.registration.unregister('transport_challenge_MA')
# except:
#     pass  # Environment was not registered
gym.envs.registration.register(
    id='transport_challenge_MA',
    entry_point='tdw-gym.tdw_gym:TDW'
)

class Challenge:
    def __init__(self, logger, port, data_path, output_dir, number_of_agents = 2, max_frames = 3000, screen_size = 256, data_prefix = 'dataset/nips_dataset/', gt_mask = False, gt_behavior = False, no_save_img = False, smart_help = False, behaviour_data_gen = False, only_save_rgb = False, seed_num = 0, oracle = False):
        self.env = TDW(port = port, number_of_agents = number_of_agents, save_dir = output_dir, max_frames = max_frames, screen_size = screen_size, data_prefix = data_prefix, gt_mask=gt_mask, gt_behavior=gt_behavior, no_save_img = no_save_img, behaviour_data_gen = behaviour_data_gen, only_save_rgb = only_save_rgb, oracle = oracle)
        self.oracle = oracle
        self.logger = logger
        self.logger.debug(port)
        self.logger.info("Environment Created")
        self.output_dir = output_dir
        self.max_frames = max_frames
        self.data = json.load(open(os.path.join(data_prefix, data_path), "r"))
        self.logger.info("done")
        self.no_save_img = no_save_img
        self.smart_help = smart_help
        self.possible_target_object = []
        self.seed_num = seed_num
        with open(os.path.join(data_prefix, 'list.json'), 'r') as f:
            mdata = json.load(f)
            if 'targets' in mdata:
                self.possible_target_object = mdata['targets']
            else:
                self.possible_target_object = mdata['food']['target'] + mdata['stuff']['target']

    def get_semantic_obs(self, state):
        agent_info = None
        objects = []
        for obj in state["1"]["visible_objects"]:
            if obj["id"] is None:
                continue
            if obj["type"] == 3 and obj["id"] == 0:
                agent_info = dict()
                agent_info["position"] = list(map(float, list(state["1"]["oppo_pos"])))
                agent_info["forward"] = list(map(float, list(state["1"]["oppo_forward"])))
                agent_info["action"] = state["1"]["previous_action"][0]
                agent_info["status"] = state["1"]["previous_status"][0]
                agent_info["held_objects"] = state["1"]["oppo_held_objects"]
            elif obj["type"] != 3:
                cur_obj = dict()
                cur_obj["name"] = obj["name"]
                if obj["name"] in self.mass_map:
                    cur_obj["weight"] = self.mass_map[obj["name"]]
                else:
                    cur_obj["weight"] = 10

                position = list(self.env.controller.object_manager.transforms[obj["id"]].position)
                cur_obj["position"] = list(map(float, position))
                cur_obj["height"] = float(position[1])
                objects.append(cur_obj)
        
        return {"agent": agent_info, "objects": objects}

    def submit(self, agents: List[PlanAgent], eval_episodes):
        total_finish = 0.0
        total_frames = 0.0
        total_runaway = 0.0
        if eval_episodes[0] == -1:
            # -1 means all tasks
            eval_episodes = range(0, len(self.data))
        num_eval_episodes = len(eval_episodes)

        start = time.time()
        results = {}
        for i, episode in enumerate(eval_episodes):
            start_time = time.time()
            if os.path.exists(os.path.join(self.output_dir, str(episode), 'result_episode.json')):
                with open(os.path.join(self.output_dir, str(episode), 'result_episode.json'), 'r') as f:
                    result = json.load(f)
                total_finish += result['finish'] / result['total']
                total_frames += result['frames']
                if 'run_away_frames' in result:
                    total_runaway += result['run_away_frames']
                results[episode] = result
                continue
            if os.path.exists(os.path.join(self.output_dir, str(episode))):
                shutil.rmtree(os.path.join(self.output_dir, str(episode)))
            # The episode has been evaluated before
            if not os.path.exists(os.path.join(self.output_dir, str(episode))):
                os.makedirs(os.path.join(self.output_dir, str(episode)))
            self.logger.info('Episode: {}/{}'.format(i + 1, num_eval_episodes))
            self.logger.info(f"Resetting Environment ... data is {self.data[episode]}")
            options = self.data[episode]
            options["possible_target_object"] = self.possible_target_object
            state, info, env_api = self.env.reset(seed=self.data[episode]['seed'] + 123456 * self.seed_num, options=options, output_dir = os.path.join(self.output_dir, str(episode)))
            for id, agent in enumerate(agents):
                if id == 0 or self.oracle:
                    #The constraint agent can know the exact target names
                    target_object_names = info['target_object_names']
                    if not ('outdoor_furniture' in self.data[0]["task"]["task_kind"]):
                        target_object_names = info['target_object_count']
                else:
                    #The helper agent only know the a set of target names
                    target_object_names = info['possible_target_object_names']
                #    if not ('outdoor_furniture' in self.data[0]["task"]["task_kind"]):
                #        target_object_names = info['possible_target_object_count']
                
                if info['goal_description'] is not None:
                    if agent.agent_type in ['plan_agent', 'follow_agent']:
                        agent.reset(obs = state[str(id)], target_object_names = target_object_names, output_dir = os.path.join(self.output_dir, str(episode)), \
                                    env_api = env_api[id], agent_color = info['agent_colors'][id], rooms_name=info['rooms_name'], info = info, obstacle_names = info['obstacle_names'], constraint_type = info['constraint'], force_ignore = self.env.force_ignore)
                    elif agent.agent_type in ['lm_agent']:
                        agent.reset(obs = state[str(id)], info = info, output_dir = os.path.join(self.output_dir, str(episode)), env_api = env_api[id], rooms_name=info['rooms_name'])
                    elif agent.agent_type in ['lm_helper_agent']:
                        agent.reset(obs = state[str(id)], info = info, output_dir = os.path.join(self.output_dir, str(episode)), env_api = env_api[id], rooms_name=info['rooms_name'])
                    elif agent.agent_type in ['child_agent', 'red_light_agent']:
                        agent.reset(obs = state[str(id)], info = info, output_dir = os.path.join(self.output_dir, str(episode)), env_api = env_api[id], seed=self.data[episode]['seed'] + 123456 * self.seed_num)
                    else:
                        raise Exception(f"{agent.agent_type} not available")
                else:
                    agent.reset(output_dir = os.path.join(self.output_dir, str(episode)))
            #for debug
            print("imgoutput", os.path.join(self.output_dir, str(episode), 'Images'))
            self.env.get_agent_api(agents)
            self.logger.info(f"Environment Reset. Took {time.time() - start_time} secs")
            local_finish = self.env.check_goal()
            done = False
            step_num = 0
            local_reward = 0.0
            if self.smart_help:
                assert len(agents) == 2, "Only support two agents for smart help data collecting"
                with open("dataset/train_dataset/outdoor_furniture/mass_map.json", "r") as f:
                    self.mass_map = json.load(f)
                obs_dict = {}
                goal_dict = {}
                plan_state_dict = {}
                constaint_type = self.data[episode]["task"]["constraint_type"]
                if constaint_type == "high":
                    height = np.random.uniform(1.5, 2.4)
                    self.env.controller.replicants[0].ability.HIGHEST_PICKUP_HEIGHT = height
                    self.env.controller.replicants[0].ability.HIGHEST_PUT_ON_HEIGHT = height
                    constraint = [(height - 1.50) / 1.50, 1.00, 1.00, 1.00, 1.00] # high, low, weight, wheelchair, bike
                elif constaint_type == "low":
                    height = np.random.uniform(0.1, 0.25)
                    self.env.controller.replicants[0].ability.LOWEST_PICKUP_HEIGHT = height
                    self.env.controller.replicants[0].ability.LOWEST_PUT_ON_HEIGHT = height
                    constraint = [1.00, (0.25 - height) / 0.25, 1.00, 1.00, 1.00]
                elif constaint_type == "weight":
                    weight = np.random.uniform(100, 300)
                    self.env.controller.replicants[0].ability.HIGHEST_PICKUP_MASS = weight
                    constraint = [1.00, 1.00, (weight - 100) / 500, 1.00, 1.00]
                elif constaint_type == "wheelchair":
                    height_high = np.random.uniform(1.5, 2.4)
                    height_low = np.random.uniform(0.1, 0.25)
                    weight = np.random.uniform(300, 500)
                    constraint = [(height_high - 1.50) / 1.50, (0.25 - height_low) / 0.25, (weight - 100) / 500, 0.00, 1.00]
                else:
                    constraint = [1.00, 1.00, 1.00, 1.00, 0.00]
                
                semantic_obs = self.get_semantic_obs(state)
                obs_dict[0] = semantic_obs

            while not done:
                step_num += 1
                #for debug, since no money for LLM to run.
                # if step_num > 10:
                    # return 0
                actions = {}
                agent_time = time.time()
                for agent_id, agent in enumerate(agents):
                    if state[str(agent_id)]['status'] != 0:
                        actions[str(agent_id)] = agent.act(state[str(agent_id)])
                    else:
                        if agent.plan_mode == "smart_help":
                            agent.update_smart_help_obs(state[str(agent_id)])

                        actions[str(agent_id)] = {"type": "ongoing"}
                self.logger.info(f"Agent step time: {time.time() - agent_time}")
                env_time = time.time()
                state, reward, done, info = self.env.step(actions)
                self.logger.info(f"Environment step time: {time.time() - env_time}")
                current_frames = state['0']['current_frames'] if 'current_frames' in state['0'] else state['1']['current_frames']
                local_reward += reward
                local_finish = self.env.check_goal()
                self.logger.info(f"Executing step {step_num} for episode: {episode}, actions: {actions}, finish: {local_finish}, frame: {self.env.num_frames}")
                # with open("file.txt", "a") as f:
                #     f.write(f"agent 0 plan for frame {self.env.num_frames}: \n")
                #     if agents[0].plan is not None:
                #         f.write(f"{agents[0].plan.value} ")
                #         target_id = agents[0].target_id
                #         if target_id is not None:
                #             f.write(f"{self.env.controller.object_manager.objects_static[target_id].name}\n")
                #         else:
                #             f.write("None\n")
                #     else:
                #         f.write("None\n")
                        
                if self.smart_help and state["1"]["status"] != 0:
                    semantic_obs = self.get_semantic_obs(state)
                    obs_dict[self.env.num_frames] = semantic_obs
                    plan_state_dict[self.env.num_frames] = agents[1].planned
                    if agents[0].plan is None:
                        goal_dict[self.env.num_frames] = [None, None, None]
                    elif agents[0].plan.value == 4:
                        goal_dict[self.env.num_frames] = ["explore", None, None]
                    elif agents[0].plan.value == 5:
                        goal_dict[self.env.num_frames] = ["wait", None, None]
                    elif agents[0].plan.value == 0 or agents[0].plan.value == 1:
                        target_id = agents[0].target_id
                        if target_id is None:
                            goal_dict[self.env.num_frames] = [None, None, None]
                        else:
                            goal_dict[self.env.num_frames] = ["pick", self.env.controller.object_manager.objects_static[target_id].name, None]
                    elif agents[0].plan.value == 2:
                        target_id = agents[0].target_id
                        if target_id is None:
                            goal_dict[self.env.num_frames] = [None, None, None]
                        else:
                            goal_dict[self.env.num_frames] = ["puton", self.env.controller.object_manager.objects_static[target_id].name, None]
                    elif agents[0].plan.value == 3:
                        # objs = state["0"]["held_objects"]
                        # if objs[0]["type"] == 1 and objs[1]["type"] == 0:
                        #     goal_dict[self.env.num_frames] = ["putin", self.env.controller.object_manager.objects_static[objs[1]["id"]].name, self.env.controller.object_manager.objects_static[objs[0]["id"]].name]
                        # elif objs[0]["type"] == 0 and objs[1]["type"] == 1:
                        #     goal_dict[self.env.num_frames] = ["putin", self.env.controller.object_manager.objects_static[objs[0]["id"]].name, self.env.controller.object_manager.objects_static[objs[1]["id"]].name]
                        # else:
                        #     goal_dict[self.env.num_frames] = [None, None, None]
                        goal_dict[self.env.num_frames] = ["putin", None, None]
                    else:
                        goal_dict[self.env.num_frames] = [None, None, None]

                if done:
                    total_frames += min(self.env.num_frames, self.max_frames)
                    break

            total_finish += local_finish[0] / local_finish[1]
            if len(agents) == 3:
                run_away_frames = agents[2].run_away_frames
                if agents[2].condition.value == 1:
                    run_away_frames += (state["2"]["current_frames"] - agents[2].last_proc_frame)
            
                result = {
                    "finish": local_finish[0],
                    "total": local_finish[1],
                    "frames": min(self.env.num_frames, self.max_frames),
                    "run_away_frames": run_away_frames
                }
                total_runaway += run_away_frames
            else:
                result = {
                    "finish": local_finish[0],
                    "total": local_finish[1],
                    "frames": min(self.env.num_frames, self.max_frames)
                }
            with open(os.path.join(self.output_dir, str(episode), 'result_episode.json'), 'w') as f:
                json.dump(result, f)
            results[episode] = result

            if self.smart_help:
                data_episode = {
                    "obs": obs_dict,
                    "goal": goal_dict,
                    "plan_state": plan_state_dict,
                    "constraint": constraint
                }
                with open(os.path.join(self.output_dir, str(episode), 'smart_help_data.json'), 'w') as f:
                    json.dump(data_episode, f)
                

        avg_finish = total_finish / num_eval_episodes
        avg_frames = total_frames / num_eval_episodes
        if len(agents) == 3:
            results = {
                "episode_results": results,
                "avg_finish": avg_finish,
                "avg_frames": avg_frames,
                "avg_emergency_rate": total_runaway / total_frames
            }
        else:
            results = {
                "episode_results": results,
                "avg_finish": avg_finish,
                "avg_frames": avg_frames
            }
        with open(os.path.join(self.output_dir, 'eval_result.json'), 'w') as f:
            json.dump(results, f)
        self.logger.info(f'eval done, avg transport rate {avg_finish}')
        self.logger.info('time: {}'.format(time.time() - start))
        return avg_finish

    def close(self):
        self.env.close()

def init_logs(output_dir, name = 'simples_example') -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(os.path.join(output_dir, "output.log"))
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="results")
    parser.add_argument("--experiment_name", type = str, default = "try")
    parser.add_argument("--run_id", type=str, default=0)
    parser.add_argument("--data_path", type=str, default="test_env.json")
    parser.add_argument("--data_prefix", type=str, default="dataset/arxiv_dataset_v3/")
    parser.add_argument("--port", default=1071, type=int)
    parser.add_argument("--agents", nargs= '+', type=str, default=("plan_agent",))
    parser.add_argument("--plan_mode", nargs= '+', type=str, default=("default",))
    parser.add_argument("--eval_episodes", nargs='+', default=(-1,), type=int, help="which episodes to evaluate on")
    parser.add_argument("--max_frames", default=3000, type=int, help="max frames per episode")
    parser.add_argument("--gt_mask", action='store_true', default=False, help="use ground truth mask")
    parser.add_argument("--gt_behavior", action='store_true', default=False, help="use ground truth behavior")
    parser.add_argument("--communication", action='store_true')
    parser.add_argument("--debug", action='store_true')
    parser.add_argument("--no_save_img", action='store_true', help="do not save images", default=False)
    parser.add_argument("--only_save_rgb", action='store_true', help="only save rgb images", default=False)
    parser.add_argument("--oracle", action='store_true', help="use oracle observation", default=False)
    parser.add_argument("--rm_behavior", action='store_true', default=False)
    # LLM parameters
    parser.add_argument('--source', default='openai',
        choices=['huggingface', 'openai'],
        help='openai API or load huggingface models')
    parser.add_argument('--lm_id', default='gpt-3.5-turbo',
                        help='name for openai engine or huggingface model name/path')
    parser.add_argument('--vlm_id', default="gpt-4o")
    parser.add_argument('--prompt_template_path', default=None, help='path to prompt template file')
    parser.add_argument('--vlm_prompt_template_path', default=None, help='path to prompt template file')
    parser.add_argument("--t", default=0.7, type=float)
    parser.add_argument("--top_p", default=1.0, type=float)
    parser.add_argument("--max_tokens", default=64, type=int)
    parser.add_argument("--n", default=1, type=int)
    parser.add_argument("--cot", action='store_true', help="use chain-of-thought prompt")
    # parser.add_argument("--echo", action='store_true', help="to include prompt in the outputs")
    parser.add_argument("--screen_size", default=256, type=int)
    parser.add_argument("--smart_help", action='store_true', default=False, help = "smart help data collecting")
    parser.add_argument("--seed_num", default=0, help = "True seed = seed_num * 123456 + seed in test env", type=int)
    args = parser.parse_args()
    args.number_of_agents = len(args.agents)
    #use os to remove non-empty dir
    os.makedirs(args.output_dir, exist_ok = True)
    args.output_dir = os.path.join(args.output_dir, args.experiment_name)
    os.makedirs(args.output_dir, exist_ok = True)
    args.output_dir = os.path.join(args.output_dir, args.run_id)
    os.makedirs(args.output_dir, exist_ok = True)
    logger = init_logs(args.output_dir)
    data = json.load(open(os.path.join(args.data_prefix, args.data_path), "r"))
    behaviour_data_gen = False
    if len(args.agents) == 2 and (args.agents[1] == 'follow_agent' or args.smart_help):
        behaviour_data_gen = True

    if "task" in data[0] and "task_kind" in data[0]["task"] and "outdoor_furniture" in data[0]["task"]["task_kind"] and not behaviour_data_gen:
        args.max_frames = 1500

    args.gt_mask = args.gt_mask or args.oracle
    args.gt_behavior = args.gt_behavior or args.oracle
    challenge = Challenge(logger, args.port, args.data_path, args.output_dir, args.number_of_agents, args.max_frames, screen_size=args.screen_size, data_prefix=args.data_prefix, gt_mask = args.gt_mask, gt_behavior = args.gt_behavior, no_save_img = args.no_save_img, smart_help = args.smart_help, behaviour_data_gen = behaviour_data_gen, only_save_rgb = args.only_save_rgb, seed_num = args.seed_num, oracle = args.oracle)
    agents = []
    task_data = challenge.data[0]
    if args.no_save_img or args.only_save_rgb:
        agent_no_save_img = True
    else:
        agent_no_save_img = False
    for i, agent in enumerate(args.agents):
        if agent == 'plan_agent':
            if task_data["task"]["task_kind"] == "outdoor_shopping":
                if i == 0:
                    agents.append(PlanAgentForBikeAgent(i, logger, args.max_frames, args, output_dir = args.output_dir, debug = args.debug, plan_mode = args.plan_mode[i], gt_mask=True, gt_behavior = True, no_save_img = agent_no_save_img, task_data = task_data))
                else:
                    agents.append(PlanAgent(i, logger, args.max_frames, args, output_dir = args.output_dir, debug = args.debug, plan_mode = args.plan_mode[i], gt_mask=args.gt_mask, gt_behavior = args.gt_behavior, no_save_img = agent_no_save_img, task_data = task_data, number_of_agents = len(args.agents), rm_behavior = args.rm_behavior))
            elif task_data["task"]["task_kind"] == "outdoor_furniture" or task_data["task"]["task_kind"] == "outdoor_furniture_crossway":
                if i == 0:
                    agents.append(PlanAgentForFurnitureAgent(i, logger, args.max_frames, args, output_dir = args.output_dir, debug = args.debug, plan_mode = args.plan_mode[i], gt_mask=True, gt_behavior = True, no_save_img = agent_no_save_img, task_data = task_data, number_of_agents = len(args.agents)))
                else:
                    agents.append(PlanAgentForFurnitureAgent(i, logger, args.max_frames, args, output_dir = args.output_dir, debug = args.debug, plan_mode = args.plan_mode[i], gt_mask=args.gt_mask, gt_behavior = args.gt_behavior, no_save_img = agent_no_save_img, task_data = task_data, number_of_agents = len(args.agents), is_oracle = args.oracle, rm_behavior = args.rm_behavior))
            else:
                if i == 0:
                    agents.append(PlanAgent(i, logger, args.max_frames, args, output_dir = args.output_dir, debug = args.debug, plan_mode = args.plan_mode[i], gt_mask=True, gt_behavior = True, no_save_img = agent_no_save_img, task_data = task_data, number_of_agents = len(args.agents)))
                else:
                    agents.append(PlanAgent(i, logger, args.max_frames, args, output_dir = args.output_dir, debug = args.debug, plan_mode = args.plan_mode[i], gt_mask=args.gt_mask, gt_behavior = args.gt_behavior, no_save_img = agent_no_save_img, task_data = task_data, number_of_agents = len(args.agents)))
        elif agent in 'follow_agent':
            assert i == 1, "Follow agent should be the second agent"
            agents.append(FollowAgent(i, logger, args.max_frames, args, output_dir = args.output_dir, debug = args.debug, plan_mode = args.plan_mode[i], gt_mask=args.gt_mask, gt_behavior = args.gt_behavior, no_save_img = agent_no_save_img, task_data = task_data))
        elif agent in 'child_agent':
            assert i == 2, "Child agent should be the third agent"
            agents.append(ChildAgent(i, logger, args.max_frames, args, output_dir = args.output_dir, debug = args.debug, plan_mode = args.plan_mode[i], gt_mask=True, gt_behavior = True, no_save_img = agent_no_save_img, task_data = task_data))
        elif agent in 'red_light_agent':
            agents.append(RedLightAgent(i, logger, args.max_frames, args, output_dir = args.output_dir, debug = args.debug, plan_mode = args.plan_mode[i], gt_mask=args.gt_mask, gt_behavior = args.gt_behavior, no_save_img = agent_no_save_img, task_data = task_data))
            pass
        else:
            raise Exception(f"{agent} not available")
    challenge.submit(agents, args.eval_episodes)
    challenge.close()

if __name__ == "__main__":
    main()