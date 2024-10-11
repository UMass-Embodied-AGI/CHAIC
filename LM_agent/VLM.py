import random

import openai
import json
import os
import pandas as pd
import backoff
import re
from openai import AzureOpenAI
from copy import deepcopy
import base64


# Open the image file and encode it as a base64 string
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

class VLM:
    def __init__(
        self,
        source,  # 'huggingface' or 'openai'
        vlm_id,
        vlm_prompt_template_path,
        communication,
        cot,
        sampling_parameters,
        agent_id,
        output_dir,
        task_kind
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
        self.vlm_prompt_template_path = None if agent_id == 0 else vlm_prompt_template_path
        self.single = "single" in self.vlm_prompt_template_path
        df = pd.read_csv(self.vlm_prompt_template_path)
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
        self.vlm_id = vlm_id
        self.chat = "gpt-35-turbo" in vlm_id or "gpt-4" in vlm_id
        self.total_cost = 0

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

        def lm_engine(source, vlm_id):
            def _generate(prompt, sampling_params):
                usage = 0
                if source == "openai":
                    try:
                        if self.chat:
                            response = client.chat.completions.create(
                                model=vlm_id, 
                                messages=prompt, 
                                **sampling_params
                            )
                            
                            dumpresponse = {}
                            dumpresponse["prompt"] = prompt[0]["content"]
                            dumpresponse["answer"] = response.choices[0].message.content
                            if self.debug:
                                with open(os.path.join(self.output_dir, "chat_raw.json"), "a") as f:
                                    f.write(json.dumps(dumpresponse, indent=4))
                                    f.write("\n")
                            generated_samples = [
                                response.choices[0].message.content for i in range(sampling_params["n"])
                            ]
                            usage = (
                                response.usage.prompt_tokens * 0.03 / 1000
                                + response.usage.completion_tokens * 0.06 / 1000
                            )
                        else:
                            raise ValueError(f"{vlm_id} not available!")
                    except Exception as e:
                        print(e)
                        raise e
                else:
                    raise ValueError("invalid source")
                # generated_samples = [sample.strip().lower() for sample in generated_samples]
                return generated_samples, usage

            return _generate

        self.generator = lm_engine(self.source, self.vlm_id)

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

        return s

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
        child_pos,
        num_previous_images_needed = 10,
    ):
        #! ================================== Prepare images ==================================
        img_abs_path_list = []
        # Get absolution path of output_dir
        output_dir_abs = os.path.abspath(self.output_dir)
        imgs_dir = os.path.join(output_dir_abs, "Images", str(self.agent_id))
        assert os.path.exists(imgs_dir) or current_step <= 0
        # count some images back, starting from "current_step", and save their absolute paths to the list
        num_appended = 0
        while num_appended < num_previous_images_needed:
            if current_step - num_appended - 1 < 0:
                break
            
            img_abs_path = os.path.join(imgs_dir, '{:04d}'.format(current_step - 1 - num_appended) + ".png")  # the file id is in 4 digits
            if os.path.exists(img_abs_path):
                img_abs_path_list.append(img_abs_path)
                num_appended += 1
                print(f"image {img_abs_path} cached.")
            else:
                break
        img_abs_path_list = img_abs_path_list[::-1]
        #! ====================================================================================
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
            assert False
            prompt += "Please choose one option from the list. "
            prompt += "Let's think step by step. "
            if self.debug:
                print(f"==> cot_prompt:\n{prompt}")
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
                {"role": "user", "content": f"According to your reasoning, please choose an option from the action list: {str(available_plans_list)}. So your choice is: "},
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
        else:
            prompt += "Please only choose one option from the list as your output and do not add any other characters. Your choice is: "
            if self.debug:
                print(f"==> base_prompt:\n{prompt}")
                
            n_trials = 0
            while n_trials < 3:
                try:
                    content = [{"type": "text", "text": prompt}]
                    for img_abs_path in img_abs_path_list:
                        content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encode_image(img_abs_path)}"}})
                    outputs, usage = self.generator(
                        [
                            # {"role": "system", "content": system_prompt},
                            {
                                "role": "user", 
                                "content": content
                            }
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
