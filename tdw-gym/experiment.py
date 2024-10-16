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
import importlib

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
import transport_challenge_multi_agent.utils as utils

gym.envs.registration.register(
    id='transport_challenge_MA',
    entry_point='tdw-gym.tdw_gym:TDW'
)

class Challenge:
    def __init__(self, logger, port, data_path, output_dir, number_of_agents = 2, max_frames = 3000, screen_size = 256, data_prefix = 'dataset/nips_dataset/', gt_mask = False, gt_behavior = False, no_save_img = False, only_save_rgb = False, seed_num = 0):
        self.env = TDW(port = port, number_of_agents = number_of_agents, save_dir = output_dir, max_frames = max_frames, screen_size = screen_size, data_prefix = data_prefix, gt_mask=gt_mask, gt_behavior=gt_behavior, no_save_img = no_save_img, only_save_rgb = only_save_rgb)
        self.logger = logger
        self.logger.debug(port)
        self.logger.info("Environment Created")
        self.output_dir = output_dir
        self.max_frames = max_frames
        self.data = json.load(open(os.path.join(data_prefix, data_path), "r"))
        self.logger.info("done")
        self.no_save_img = no_save_img
        self.possible_target_object = []
        self.seed_num = seed_num
        with open(os.path.join(data_prefix, 'list.json'), 'r') as f:
            mdata = json.load(f)
            if 'targets' in mdata:
                self.possible_target_object = mdata['targets']
            else:
                self.possible_target_object = mdata['food']['target'] + mdata['stuff']['target']

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
            if not os.path.exists(os.path.join(self.output_dir, str(episode))):
                os.makedirs(os.path.join(self.output_dir, str(episode)))
                
            self.logger.info('Episode: {}/{}'.format(i + 1, num_eval_episodes))
            self.logger.info(f"Resetting Environment ... data is {self.data[episode]}")
            options = self.data[episode]
            options["possible_target_object"] = self.possible_target_object
            state, info, env_api = self.env.reset(seed=self.data[episode]['seed'] + 123456 * self.seed_num, options=options, output_dir = os.path.join(self.output_dir, str(episode)))
            
            for id, agent in enumerate(agents):
                if id == 0:
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
                    if id == 1:
                        agent.reset(obs = state[str(id)], info = info)
                    else:
                        if agent.agent_type == 'plan_agent':
                            agent.reset(obs = state[str(id)], target_object_names = target_object_names, output_dir = os.path.join(self.output_dir, str(episode)), \
                                        env_api = env_api[id], agent_color = info['agent_colors'][id], rooms_name=info['rooms_name'], info = info, obstacle_names = info['obstacle_names'], constraint_type = info['constraint'], force_ignore = self.env.force_ignore)
                        elif agent.agent_type == 'child_agent':
                            agent.reset(obs = state[str(id)], info = info, output_dir = os.path.join(self.output_dir, str(episode)), env_api = env_api[id], seed=self.data[episode]['seed'] + 123456 * self.seed_num)
                        else:
                            raise Exception(f"{agent.agent_type} not available")
                else:
                    if id == 1:
                        agent.reset(obs = state[str(id)], info = info)
                    else:
                        agent.reset(output_dir = os.path.join(self.output_dir, str(episode)))
            #for debug
            # print("imgoutput", os.path.join(self.output_dir, str(episode), 'Images'))
            self.env.get_agent_api(agents)
            self.logger.info(f"Environment Reset. Took {time.time() - start_time} secs")
            local_finish = self.env.check_goal()
            done = False
            step_num = 0
            local_reward = 0.0
            
            while not done:
                step_num += 1
                actions = {}
                agent_time = time.time()
                for agent_id, agent in enumerate(agents):
                    if state[str(agent_id)]['status'] != 0:
                        actions[str(agent_id)] = agent.act(state[str(agent_id)])
                    else:
                        actions[str(agent_id)] = {"type": "ongoing"}
                        
                self.logger.info(f"Agent step time: {time.time() - agent_time}")
                env_time = time.time()
                state, reward, done, info = self.env.step(actions)
                self.logger.info(f"Environment step time: {time.time() - env_time}")
                local_reward += reward
                local_finish = self.env.check_goal()
                self.logger.info(f"Executing step {step_num} for episode: {episode}, actions: {actions}, finish: {local_finish}, frame: {self.env.num_frames}")
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
    # set plan mode for each agent to default
    # parser.add_argument("--plan_mode", nargs= '+', type=str, default=("default",))
    parser.add_argument("--eval_episodes", nargs='+', default=(-1,), type=int, help="which episodes to evaluate on")
    parser.add_argument("--max_frames", default=3000, type=int, help="max frames per episode")
    parser.add_argument("--gt_mask", action='store_true', default=False, help="use ground truth mask")
    parser.add_argument("--gt_behavior", action='store_true', default=False, help="use ground truth behavior")
    parser.add_argument("--debug", action='store_true')
    parser.add_argument("--no_save_img", action='store_true', help="do not save images", default=False)
    parser.add_argument("--only_save_rgb", action='store_true', help="only save rgb images", default=False)
    # parser.add_argument("--rm_behavior", action='store_true', default=False)
    parser.add_argument("--screen_size", default=256, type=int)
    parser.add_argument("--seed_num", default=0, help = "True seed = seed_num * 123456 + seed in test env", type=int)
    
    args = parser.parse_args()
    args.number_of_agents = len(args.agents)
    #use os to remove non-empty dir
    os.makedirs(args.output_dir, exist_ok = True)
    args.output_dir = os.path.join(args.output_dir, args.experiment_name)
    os.makedirs(args.output_dir, exist_ok = True)
    args.output_dir = os.path.join(args.output_dir, args.run_id)
    os.makedirs(args.output_dir, exist_ok = True)
    args.logger = init_logs(args.output_dir)
    data = json.load(open(os.path.join(args.data_prefix, args.data_path), "r"))
    if "task" in data[0] and "task_kind" in data[0]["task"] and "outdoor_furniture" in data[0]["task"]["task_kind"]:
        args.max_frames = 1500 # set max frames of outdoor furniture task to 1500

    challenge = Challenge(args.logger, args.port, args.data_path, args.output_dir, args.number_of_agents, args.max_frames, screen_size=args.screen_size, data_prefix=args.data_prefix, gt_mask = args.gt_mask, gt_behavior = args.gt_behavior, no_save_img = args.no_save_img, only_save_rgb = args.only_save_rgb, seed_num = args.seed_num)
    agents = []
    args.task_data = challenge.data[0]
    if args.no_save_img or args.only_save_rgb:
        args.agent_no_save_img = True
    else:
        args.agent_no_save_img = False
        
    args.plan_mode = ["default"] * len(args.agents)
        
    for i, agent in enumerate(args.agents):
        if i != 1:
            if agent == 'plan_agent':
                if args.task_data["task"]["task_kind"] == "outdoor_shopping":
                    agents.append(PlanAgentForBikeAgent(i, args.logger, args.max_frames, args, output_dir = args.output_dir, debug = args.debug, plan_mode = args.plan_mode[i], gt_mask=True, gt_behavior = True, no_save_img = args.agent_no_save_img, task_data = args.task_data))
                elif args.task_data["task"]["task_kind"] == "outdoor_furniture" or args.task_data["task"]["task_kind"] == "outdoor_furniture_crossway":
                    agents.append(PlanAgentForFurnitureAgent(i, args.logger, args.max_frames, args, output_dir = args.output_dir, debug = args.debug, plan_mode = args.plan_mode[i], gt_mask=True, gt_behavior = True, no_save_img = args.agent_no_save_img, task_data = args.task_data, number_of_agents = len(args.agents)))
                else:
                    agents.append(PlanAgent(i, args.logger, args.max_frames, args, output_dir = args.output_dir, debug = args.debug, plan_mode = args.plan_mode[i], gt_mask=True, gt_behavior = True, no_save_img = args.agent_no_save_img, task_data = args.task_data, number_of_agents = len(args.agents)))
            elif agent in 'child_agent':
                assert i == 2, "Child agent should be the third agent"
                agents.append(ChildAgent(i, args.logger, args.max_frames, args, output_dir = args.output_dir, debug = args.debug, plan_mode = args.plan_mode[i], gt_mask=True, gt_behavior = True, no_save_img = args.agent_no_save_img, task_data = args.task_data))
            else:
                raise Exception(f"{agent} not available")
        else:
            agent_path = f"agent/{agent}.py"
            if not os.path.exists(agent_path):
                raise Exception(f"Agent {agent} not found")
            
            module = importlib.import_module(f"agent.{agent}", package=None)
            cls = getattr(module, "PlanAgent", None)
            agents.append(cls(i, args)) # modify here to pass in the correct arguments
            
    challenge.submit(agents, args.eval_episodes)
    challenge.close()

if __name__ == "__main__":
    main()