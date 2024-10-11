import random

import openai
import json
import os
import pandas as pd
# from openai.error import OpenAIError
# from openai import OpenAI
import backoff
import re
from openai import AzureOpenAI
from copy import deepcopy


class LLM:
    def __init__(
        self,
        source,  # 'huggingface' or 'openai'
        lm_id,
        prompt_template_path,
        communication,
        cot,
        sampling_parameters,
        agent_id,
        output_dir,
        task_kind,
        rm_behavior
    ):
        self.output_dir = output_dir
        self.rooms_explored = None
        self.goal_desc = None
        self.agent_id = agent_id
        self.task_kind = task_kind
        if "high" in self.task_kind or "furniture" in self.task_kind:
            self.agent_name = "Alice" if agent_id == 0 else "Bob"
            self.oppo_name = "Alice" if agent_id == 1 else "Bob"
            self.oppo_pronoun = "she" if agent_id == 1 else "he"
        else:
            self.agent_name = "David" if agent_id == 0 else "Bob"
            self.oppo_name = "David" if agent_id == 1 else "Bob"
            self.oppo_pronoun = "he"

        self.debug = sampling_parameters.debug
        self.rooms = []
        # self.prompt_template_path = "./LLM_agent/modified_prompts/prompt_single.csv" if agent_id == 0 else "./LLM_agent/modified_prompts/prompt_helper_obstacle.csv"
        # self.prompt_template_path = "./LLM_agent/prompt_single.csv" if agent_id == 0 else "./LLM_agent/prompt_helper.csv"
        self.prompt_template_path = None if agent_id == 0 else prompt_template_path
        self.single = "single" in self.prompt_template_path
        df = pd.read_csv(self.prompt_template_path)
        # self.system_prompt_template = (
        #     df["system prompt"][0].replace("$AGENT_NAME$", self.agent_name).replace("$OPPO_NAME$", self.oppo_name)
        # )
        self.prompt_template = (
            df["prompt"][0].replace("$AGENT_NAME$", self.agent_name).replace("$OPPO_NAME$", self.oppo_name)
        )
        # if communication:
        #     self.generator_prompt_template = (
        #         df["prompt"][1].replace("$AGENT_NAME$", self.agent_name).replace("$OPPO_NAME$", self.oppo_name)
        #     )
        # else:
        #     self.generator_prompt_template = None

        self.communication = communication
        self.cot = cot
        self.source = source
        self.lm_id = lm_id
        self.chat = "gpt-35-turbo" in lm_id or "gpt-4" in lm_id
        self.total_cost = 0
        self.rm_behavior = rm_behavior

        if self.source == "openai":
            #######################################################
            client = AzureOpenAI(
                azure_endpoint = "Your Azure Endpoint Here", 
                api_key="Your API Key Here",  
                api_version="2024-02-15-preview"
            )
            #######################################################
            if self.chat:
                self.sampling_params = {
                    "max_tokens": sampling_parameters.max_tokens,
                    "temperature": sampling_parameters.t,
                    "top_p": sampling_parameters.top_p,
                    "n": sampling_parameters.n,
                }
            else:
                self.sampling_params = {
                    "max_tokens": sampling_parameters.max_tokens,
                    "temperature": sampling_parameters.t,
                    "top_p": sampling_parameters.top_p,
                    "n": sampling_parameters.n,
                    "logprobs": sampling_parameters.logprobs,
                    "echo": sampling_parameters.echo,
                }
        else:
            raise ValueError("invalid source")

        def lm_engine(source, lm_id):
            def _generate(prompt, sampling_params):
                usage = 0
                if source == "openai":
                    try:
                        if self.chat:
                            response = client.chat.completions.create(
                                model="gpt4" if lm_id == "gpt-4" else lm_id, 
                                messages=prompt, 
                                **sampling_params
                            )
                            dumpresponse = {}
                            cur_prompt = ""
                            for i in range(len(prompt)):
                                cur_prompt += f"{prompt[i]['role']}: {prompt[i]['content']}\n"
                                
                            dumpresponse["prompt"] = cur_prompt
                            dumpresponse["answer"] = response.choices[0].message.content
                            if self.debug:
                                with open(os.path.join(self.output_dir, "chat_raw.json"), "a") as f:
                                    f.write(json.dumps(dumpresponse, indent=4))
                                    f.write("\n")
                            generated_samples = [
                                response.choices[0].message.content for i in range(sampling_params["n"])
                            ]
                            if "gpt-4" in self.lm_id:
                                usage = (
                                    response.usage.prompt_tokens * 0.03 / 1000
                                    + response.usage.completion_tokens * 0.06 / 1000
                                )
                            elif "gpt-35-turbo" in self.lm_id:
                                usage = response.usage.total_tokens * 0.002 / 1000
                        # mean_log_probs = [np.mean(response['choices'][i]['logprobs']['token_logprobs']) for i in
                        #                   range(sampling_params['n'])]
                        elif "text-" in lm_id:
                            raise NotImplementedError
                            response = openai.Completion.create(model=lm_id, prompt=prompt, **sampling_params)
                            # print(json.dumps(response, indent=4))
                            if self.debug:
                                with open(f"LLM/raw.json", "a") as f:
                                    f.write(json.dumps(response, indent=4))
                                    f.write("\n")
                            generated_samples = [response.choices[i].text for i in range(sampling_params["n"])]
                        # mean_log_probs = [np.mean(response['choices'][i]['logprobs']['token_logprobs']) for i in
                        #               range(sampling_params['n'])]
                        else:
                            raise ValueError(f"{lm_id} not available!")
                    except Exception as e:
                        print(e)
                        raise e
                else:
                    raise ValueError("invalid source")
                # generated_samples = [sample.strip().lower() for sample in generated_samples]
                return generated_samples, usage

            return _generate

        self.generator = lm_engine(self.source, self.lm_id)

        self.current_room = None
        self.object_list = None
        self.holding_objects = None

    def reset(self, rooms_name, info, output_dir):
        self.output_dir = output_dir
        self.rooms = rooms_name
        self.info = info
        self.goal_position_names = info["goal_position_names"]
        self.names_mapping = info["names_mapping"]
        self.goal_position_id = info["goal_position_id"]
        self.goal_position_name = info["goal_position_name"]
        self.goal_position_desc = f"<{self.goal_position_name}> ({self.goal_position_id})"
        self.constraint = info["constraint"]
        goal_objects = info["goal_description"].keys()
        self.goal_desc = self.goal2description(goal_objects)

    def goal2description(self, goals):  # {predicate: count}
        s = "Transport "
        s += f"all {goals}s, "

        s = s[:-2] + f" to the {self.goal_position_desc}."
        return s
    
    # def possible_task2description(self, possible_task, goal_position_names):
    #     if isinstance(goal_position_names, str):
    #         goal_position_names = [goal_position_names]
            
    #     ret = []
    #     for g in goal_position_names:
    #         for p in possible_task:
    #             s = "Transport "
    #             for i, x in enumerate(p):
    #                 item, number = x[0], x[1]
    #                 if i < len(p) - 1:
    #                     s += f"{number} <{self.names_mapping[item]}>, "
    #                 else:
    #                     s += f"and {number} <{self.names_mapping[item]}>"
    #             s += f" to the <{g}>."
    #             ret.append(s)
                    
    #     return str(ret)

    def parse_answer(self, available_actions, text):
        # text = re.sub(r"^[A-Za-z]\.?\s?", "", text)
        text = text.replace("\n", "").replace("\r", "").replace(".", "")
        if text[0] == "'" and text[-1] == "'":
            text = text[1:-1]

        if "goto and pick up" in text or "remove obstacle" in text:
            try:
                text_action = text.split(" <")[0]
                text_id = text.split("(")[-1].split(")")[0]
            except:
                raise ValueError(f"Failed to parse text: {text}")
        else:
            text_action = text
            text_id = "None"

        if text_id == "None":
            text_id = None
        else:
            try:
                text_id = int(text_id)
            except:
                raise ValueError(f"Failed to parse text_id: {text_id}")
        return text_action, text_id
        # for i in range(len(available_actions)):
        #     action = available_actions[i]
        #     if action in text:
        #         return action

        # for i in range(len(available_actions)):
        #     action = available_actions[i]
        #     option = chr(ord('A') + i)
        #     # txt = text.lower()
        #     if f"option {option}" in text or f"{option}." in text.split(' ') or f"{option}," in text.split(' ') or f"Option {option}" in text or f"({option})" in text or f"action {option}" in text or (len(text) <= 2 and option in text):
        #         return action
        # print("WARNING! Fuzzy match!")
        # for i in range(len(available_actions)):
        #     action = available_actions[i]
        #     if self.communication and i == 0:
        #         continue
        #     act = "None"
        #     name = "None"
        #     id = "None"
        #     if action.startswith('go to'):
        #         # act = 'go to'
        #         name = action.split(' ')[-2][1:-1]
        #         id = action.split(' ')[-1][1:-1]
        #     elif action.startswith('explore'):
        #         act = 'explore'
        #         name = action.split(' ')[-2][1:-1]
        #         id = action.split(' ')[-1][1:-1]
        #     elif action.startswith('go grasp'):
        #         act = 'grasp'
        #         name = action.split(' ')[-2][1:-1]
        #         id = action.split(' ')[-1][1:-1]
        #     elif action.startswith('put'):
        #         act = 'put'
        #     elif action.startswith('transport'):
        #         act = 'transport'
        #     option = chr(ord('A') + i)
        #     if f"{option} " in text or act in text or name in text or id in text:
        #         return action
        # if len(text) == 1:
        #     i = ord(text) - ord('A')
        #     if i in range(len(available_actions)):
        #         return available_actions[i]
        # print("WARNING! No available action parsed!!! Random choose one")
        # return random.choice(available_actions)

    def oppo_holding(self, opponent_grabbed_objects_history):
        # s_hold = ["", ""]
        holded = []
        for opponent_grabbed_objects in opponent_grabbed_objects_history:
            for i, obj in enumerate(opponent_grabbed_objects):
                if obj["type"] == 0:
                    holded.append(f"a target object <{obj['name']}> ({obj['id']})")
                elif obj["type"] == 1:
                    ss = ""
                    cnt = 0
                    for j, o in enumerate(obj["contained"]):
                        if o is None:
                            break
                        cnt += 1
                        ss += f"<{obj['contained_name'][j]}> ({o}), "
                    if cnt == 0:
                        ss = "nothing"
                    else:
                        ss = f"target object{'s' if cnt > 1 else ''} {ss[:-2]}"
                    holded.append(f"a container <{obj['name']}> ({obj['id']}) with {ss} in it")
                elif "shopping" in self.task_kind and obj["name"] == "bike":
                    ss = ""
                    cnt = 0
                    for j, o in enumerate(obj["contained"]):
                        if o is None:
                            break
                        cnt += 1
                        ss += f"<{obj['contained_name'][j]}> ({o}), "
                    if cnt == 0:
                        ss = "nothing"
                    else:
                        ss = f"target object{'s' if cnt > 1 else ''} {ss[:-2]}"
                    holded.append(f"a bike with {ss} in it")


        if len(holded) == 0:
            return f"You haven't seen {self.oppo_name} holding any object before"
        
        unique_holded = []
        for x in holded:
            if x not in unique_holded:
                unique_holded.append(x)

        s = f"You have seen {self.oppo_name} holding these objects: "
        s += ", ".join(unique_holded)

        return s


    def progress2text(
        self,
        current_step,
        satisfied,
        opponent_grabbed_objects,
        opponent_last_room,
    ):
        if "furniture" in self.task_kind:
            s = f"You've taken {current_step}/1500 steps. "
        else:
            s = f"You've taken {current_step}/3000 steps. "

        """
        for room, obj_list in self.obj_per_room.items():
            sr = ""
            s_obj = ""
            s_con = ""
            s_bed = ""
            objs = obj_list[0]
            cons = obj_list[1]
            if len(objs) > 0:
                if len(objs) == 1:
                    x = objs[0]
                    s_obj += f"a target object <{x['name']}> ({x['id']})"
                else:
                    ss = ', '.join([f"<{x['name']}> ({x['id']})" for x in objs])
                    s_obj += f"target objects " + ss
            
            if len(cons) > 0:
                if len(cons) == 1:
                    x = cons[0]
                    s_con = f"a container <{x['name']}> ({x['id']})"
                else:
                    ss = ', '.join([f"<{x['name']}> ({x['id']})" for x in cons])
                    s_con = f"containers " + ss
            if len(obj_list[2]) > 0:
                s_bed = 'the goal position {self.goal_position_desc}'
            if s_obj == "" and s_con == "" and s_bed == "":
                sr += 'nothing'
            elif s_obj != "" and s_con != "" and s_bed == "":
                sr += s_obj + ', and ' + s_con
            elif s_obj != "" and s_con == "" and s_bed != "":
                sr += s_obj + ', and ' + s_bed
            elif s_obj == "" and s_con != "" and s_bed != "":
                sr += s_con + ', and ' + s_bed
            elif s_obj != "" and s_con != "" and s_bed != "":
                sr += s_obj + ', ' + s_con + ', and ' + s_bed
            else:
                sr += s_obj + s_con + s_bed
            sss[room] = sr
        """
        if len(self.object_list[2]) == 0:
            s += "You haven't found the goal position. "
        else:
            s += f"You have found the goal position {self.goal_position_desc}. "

        if len(satisfied) != 0:
            s += f"You and {self.oppo_name} have already transported "
            unique_satisfied = []
            for x in satisfied:
                if x not in unique_satisfied:
                    unique_satisfied.append(x)
            if len([x for x in unique_satisfied if x["type"] == 0]) == 0:
                s += "nothing"
            s += ", ".join([f"<{x['name']}> ({x['id']})" for x in unique_satisfied if x["type"] == 0])
            s += f" to the {self.goal_position_desc}. "

        s_hold = ["", ""]
        for i, obj in enumerate(self.holding_objects):
            if obj["type"] == 0:
                s_hold[i] = f"a target object <{obj['name']}> ({obj['id']})"
            elif obj["type"] == 1:
                ss = ""
                cnt = 0
                for j, o in enumerate(obj["contained"]):
                    if o is None:
                        break
                    cnt += 1
                    ss += f"<{obj['contained_name'][j]}> ({o}), "
                if cnt == 0:
                    ss = "nothing"
                else:
                    ss = f"target object{'s' if cnt > 1 else ''} {ss[:-2]}"
                
                s_hold[i] = f"a container <{obj['name']}> ({obj['id']}) with {ss} in it"

        if self.holding_objects[0]["type"] == 0 and self.holding_objects[1]["type"] == 0:
            s += f"You're holding two target objects <{self.holding_objects[0]['name']}> ({self.holding_objects[0]['id']}) and <{self.holding_objects[1]['name']}> ({self.holding_objects[1]['id']})"
        elif s_hold[0] == "" and s_hold[1] == "":
            s += "You're holding nothing"
        elif s_hold[0] != "" and s_hold[1] != "":
            s += f"You're holding {s_hold[0]}, and in another hand, {s_hold[1]}"
        else:
            s += f"You're holding {s_hold[0]}{s_hold[1]}"

        # s += "The objects that are already transported are: $SATISFIED$"

        # print(self.current_room, self.obj_per_room)
        """
        if self.current_room not in self.rooms_explored: pred_room = 'none'
        else: pred_room = self.rooms_explored[self.current_room]
        if pred_room != 'all' and sss[self.current_room] == 'nothing':
            s += f"You're in the {self.current_room}, where You've explored {pred_room} of it. "
        else:
            s += f"You're in the {self.current_room}, where You've explored {pred_room} of it and found {sss[self.current_room]}. "
        """
            
        """
            if opponent_last_room is None:
                s += f"You don't know where {self.oppo_name} is. "
            elif opponent_last_room == self.current_room:
                s += f"You also see {self.oppo_name} here in the {self.current_room}, {self.oppo_pronoun} is holding {ss}"
            else:
                s += f"Last time You saw {self.oppo_name} was in the {opponent_last_room}, {self.oppo_pronoun} was holding {ss}"

        for room in self.rooms:
            if room == self.current_room:
                continue
            #s += f"You've explored {self.rooms_explored[room] if room in self.rooms_explored else 'None'} of the {room}, and You found {sss[room]} there. "
            if room not in self.rooms_explored: pred_room = 'none'
            else: pred_room = self.rooms_explored[room]
            if pred_room != 'all' and sss[room] == 'nothing':
                s += f"You've explored {pred_room} of the {room}. "
            else:
                s += f"You've explored {pred_room} of the {room}, and You found {sss[room]} there. "
        """

        return s

    def get_available_plans(self, message):
        assert False
        """
        go to room {}
        explore current room {}
        go grasp target object / container {}
        holding both container and object: put obj into the container
        holding any goal objects: transport holding objects to the bed
        send a message: ""
        """
        available_plans = []
        if self.communication and message is not None:
            available_plans.append(f"send a message: {message}")
        if self.holding_objects[0]["type"] is None or self.holding_objects[1]["type"] is None:
            for obj in self.object_list[0]:
                available_plans.append(f"go grasp target object <{obj['name']}> ({obj['id']})")
            if not (self.holding_objects[0]["type"] == 1 or self.holding_objects[1]["type"] == 1):
                for obj in self.object_list[1]:
                    available_plans.append(f"go grasp container <{obj['name']}> ({obj['id']})")
        else:
            if (
                self.holding_objects[0]["type"] == 1
                and self.holding_objects[0]["contained"][-1] is None
                and self.holding_objects[1]["type"] == 0
            ):
                available_plans.append(
                    f"put <{self.holding_objects[1]['name']}> ({self.holding_objects[1]['id']}) into the container <{self.holding_objects[0]['name']}> ({self.holding_objects[0]['id']})"
                )
            elif (
                self.holding_objects[1]["type"] == 1
                and self.holding_objects[1]["contained"][-1] is None
                and self.holding_objects[0]["type"] == 0
            ):
                available_plans.append(
                    f"put <{self.holding_objects[0]['name']}> ({self.holding_objects[0]['id']}) into the container <{self.holding_objects[1]['name']}> ({self.holding_objects[1]['id']})"
                )
        if any(obj["type"] is not None for obj in self.holding_objects) and len(self.object_list[2]) != 0:
            available_plans.append(f"transport objects You're holding to the {self.goal_position_desc}")
        for room in self.rooms:
            if room == self.current_room or room is None or room == "None":
                continue
            available_plans.append(f"go to {room}")
        if self.current_room not in self.rooms_explored or self.rooms_explored[self.current_room] != "all":
            available_plans.append(f"explore current room {self.current_room}")

        plans = ""
        for i, plan in enumerate(available_plans):
            plans += f"{chr(ord('A') + i)}. {plan}\n"

        return plans, len(available_plans), available_plans

        # return self.LLM.run_once_split(self.num_frames, self.obs['held_objects'], self.agent_memory.object_info, self.satisfied, self.object_list, self.action_history, self.obs['oppo_held_objects'], self.prefer_target, valid_plan)

    def make_action_status_history(self, action_history, status_history):
        res = list((a.replace(" with left hand", "").replace(" with right hand", "").replace(" in the left hand", "").replace(" in the right hand", ""), s) for (a, s) in zip(action_history, status_history) if a is not None and s is not None) 

        return res
    
    def run(
        self,
        current_step,
        holding_objects,
        object_info,
        satisfied,
        object_list,
        action_history,
        status_history,
        oppo_action_history,
        oppo_status_history,
        opponent_grabbed_objects_history,
        valid_plan,
        current_pos,
        oppo_pos,
        child_pos
    ):
        info = {}
        print("current_step", current_step)
        self.holding_objects = holding_objects
        self.object_list = object_list
        
        #! =================================== Prepare prompt ===================================
        # system_prompt = self.system_prompt_template
        prompt = self.prompt_template
        prompt = prompt.replace("$CURRENT_POSITION$", f"{round(current_pos[0], 2), round(current_pos[2], 2)}")
        progress_desc = self.progress2text(current_step, satisfied, opponent_grabbed_objects_history, None)
        prompt = prompt.replace("$PROGRESS$", progress_desc)
        action_history_desc = self.make_action_status_history(action_history, status_history)
        prompt = prompt.replace("$ACTION_HISTORY$", str(action_history_desc))
        prompt = prompt.replace("$GOAL_POSITION$", str(self.goal_position_desc))
        if self.agent_id == 0:
            prompt = prompt.replace("$GOAL$", self.goal_desc)
        if self.agent_id == 1:
            prompt = prompt.replace("$CONSTRAINT$", str(self.constraint))
            if self.rm_behavior:
                pt = f"{self.oppo_name}'s previous actions and status are: $OPPO_ACTION_HISTORY$. "
                assert pt in prompt
                prompt = prompt.replace(pt, "")
            else:
                assert oppo_action_history is not None and oppo_status_history is not None
                oppo_action_history_desc = self.make_action_status_history(oppo_action_history, oppo_status_history)
                prompt = prompt.replace("$OPPO_ACTION_HISTORY$", str(oppo_action_history_desc))

            oppo_holding = self.oppo_holding(opponent_grabbed_objects_history)
            prompt = prompt.replace("$OPPO_HOLDING$", str(oppo_holding))

        translate_object_info = ", ".join(
            [
                f"{'container' if x['type'] == 1 else 'obstacle' if x['type'] == 4 else ''} <{x['name']}> ({x['id']}) is located at {round(x['position'][0], 2), round(x['position'][2], 2)} (meters) with a height of {round(x['position'][1], 2)} meters"
                for x in object_info.values()
                if x["type"] != 3
            ]
        )
        oppo_pos_desc = f"You've never seen {self.oppo_name} before"
        if oppo_pos is not None:
            oppo_pos_desc = f"The last time you saw, {self.oppo_name} was at {round(oppo_pos[0], 2), round(oppo_pos[2], 2)} (meters)"

        prompt = prompt.replace("$OPPO_POS_DESC$", str(oppo_pos_desc))

        child_pos_desc = f"You've never seen {self.oppo_name}'s child before"
        if child_pos is not None:
            child_pos_desc = f"The last time you saw, {self.oppo_name}'s child was at {round(child_pos[0], 2), round(child_pos[2], 2)} (meters)"

        prompt = prompt.replace("$CHILD_POS_DESC$", str(child_pos_desc))

        prompt = prompt.replace("$OBJECT_INFO$", str(translate_object_info) if str(translate_object_info) != "" else "nothing")
        # satisfied_object_info = ", ".join(
        #     [
        #         f"<{x['name']}> ({x['id']}) is located at {round(x['position'][0], 2), round(x['position'][2], 2)} (meters) with a height of {round(x['position'][1], 2)} meters"
        #         for x in satisfied
        #     ]
        # )
        # prompt = prompt.replace("$SATISFIED$", str(satisfied_object_info) if str(satisfied_object_info) != "" else "nothing")

        #! --- Make "available_plans" ---
        message = None
        
        # available_plans = valid_plan
        # plans = ""
        # for i, plan in enumerate(available_plans):
        #     plans += f"{chr(ord('A') + i)}. {plan}\n"
        # available_plans, num, available_plans_list = plans, len(available_plans), available_plans

        # available_plans, num, available_plans_list = self.get_available_plans(message)
        has_target, has_container, has_obstacle, has_follow = False, False, False, False
        valid_plan_copy = deepcopy(valid_plan)
        for p in valid_plan:
            if "goto and pick up target" in p:
                has_target = True
                valid_plan_copy.remove(p)
            elif "goto and pick up container" in p:
                has_container = True
                valid_plan_copy.remove(p)
            elif "remove obstacle" in p:
                has_obstacle = True
                valid_plan_copy.remove(p)
            elif "follow another agent" in p:
                has_follow = True
                valid_plan_copy.remove(p)

        available_plans_list = valid_plan_copy
        assert "explore" in available_plans_list[0]
        if has_target:
            available_plans_list.extend([f"goto and pick up target <{x['name']}> ({x['id']})" for x in object_info.values() if x["type"] == 0])
        if has_container:
            available_plans_list.extend([f"goto and pick up container <{x['name']}> ({x['id']})" for x in object_info.values() if x["type"] == 1])
        if has_obstacle:
            available_plans_list.extend([f"remove obstacle <{x['name']}> ({x['id']})" for x in object_info.values() if x["type"] == 4])
        if has_follow:
            available_plans_list.append(f"follow {self.oppo_name}")
        num = len(available_plans_list)
        
        # available_plans = ""
        # for i, plan in enumerate(available_plans_list):
        #     available_plans += f"{chr(ord('A') + i)}. {plan}\n"
        #! -----------------------------

        if num == 0 or (message is not None and num == 1):
            raise ValueError("No available plans!")
            print("Warning! No available plans!")
            plan = None
            info.update({"num_available_actions": num, "plan": None})
            
        if num == 1:
            assert available_plans_list[0] == "explore"
            return "explore", None, info

        prompt = prompt.replace("$AVAILABLE_ACTIONS$", str(available_plans_list))

        #! ======================================================================================

        if self.cot:
            prompt += "Please choose one option from the list. "
            prompt += "Let's think step by step. "
            if self.debug:
                print(f"==> cot_prompt:\n{prompt}")

            n_trials = 0
            while n_trials < 3:
                try:
                    chat_messages = [
                        # {"role": "system", "content": system_prompt}, 
                        {"role": "user", "content": prompt}
                    ]
                    outputs, usage = self.generator(chat_messages if self.chat else prompt, self.sampling_params)
                    output = outputs[0]
                    self.total_cost += usage
                    info["cot_outputs"] = outputs
                    info["cot_usage"] = usage
                    if self.debug:
                        print(f"==> cot_output:\n{output}")
                    chat_messages = [
                        # {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt},
                        {"role": "assistant", "content": output},
                        {"role": "user", "content": f"According to your reasoning, please choose an option from the action list as your output: {str(available_plans_list)}. Do not add any other characters. So your choice is: "},
                    ]
                    normal_prompt = prompt + output + " So the answer is: "
                    outputs, usage = self.generator(chat_messages if self.chat else normal_prompt, self.sampling_params)
                    output = outputs[0]
                    self.total_cost += usage
                    info["output_usage"] = usage
                    if self.debug:
                        print(f"==> base_output:\n{output}")
                        print(f"==> total cost: {self.total_cost}")
                    
                    plan, plan_id = self.parse_answer(available_plans_list, output)
                    break
                except:
                    print(f"Failed to generate a plan! Retrying...")
                    import time 
                    time.sleep(30)
                    n_trials += 1
                    continue
                
            if n_trials == 5:
                print("Failed to generate a plan!")
                plan, plan_id = "explore", "None"

        else:
            prompt += "Please only choose one option from the list as your output and do not add any other characters. Your choice is: "
            if self.debug:
                print(f"==> base_prompt:\n{prompt}")
                
            n_trials = 0
            while n_trials < 3:
                try:
                    outputs, usage = self.generator(
                        [
                            # {"role": "system", "content": system_prompt},
                            {"role": "user", "content": prompt}
                        ] if self.chat else prompt, self.sampling_params
                    )
                    output = outputs[0]
                    info["cot_usage"] = usage
                    if self.debug:
                        print(f"==> base_output:\n{output}")
                
                    plan, plan_id = self.parse_answer(available_plans_list, output)
                    break
                except:
                    print(f"Failed to generate a plan! Retrying...")
                    import time 
                    time.sleep(30)
                    n_trials += 1
                    continue
                
            if n_trials == 5:
                print("Failed to generate a plan!")
                plan, plan_id = "explore", "None"

            
        if self.debug:
            print(f"plan: {plan}\n")
        info.update(
            {
                "num_available_actions": num,
                "prompts": prompt,
                "outputs": outputs,
                "plan": plan,
                "total_cost": self.total_cost,
            }
        )
        
        return plan, plan_id, info
