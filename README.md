# Constrained Human-AI Cooperation (CHAIC): An Inclusive Embodied Social Intelligence Challenge

## ✨ Introduction
This is the anonymous raw code for the NeurIPS Dataset Track submission CHAIC.

You could view the [[Project Page](https://chaic-neurips.github.io/CHAIC/)] for some video demos.

> We introduce the Constrained Human-AI Cooperation (CHAIC), an inclusive embodied social intelligence challenge for testing social perception and cooperation in embodied agents. In CHAIC, the goal is for an embodied agent equipped with egocentric observations to aid a human possibly operating under physical constraints, e.g. unable to reach high places or confined to a wheelchair, to perform common household or outdoor tasks as efficiently as possible. To do this, a successful helper must (1). infer the human's intents and constraints by following the human and observing their behaviors (social perception), and (2). make a cooperative plan tailored to the human user to solve the task as fast as possible together as a team (cooperative planning). 
To benchmark this challenge, we created 4 new agents with real physical constraints, and 8 long-horizon tasks featuring both indoor and outdoor scenes with various constraints and emergency events along with potential risks. We benchmark both planning and learning baselines on the challenge and introduce a new method leveraging Large Language Models and behavior modeling. Empirical evaluation demonstrates the ability of our benchmark to enable systematic evaluation of important elements of machine social intelligence.

<div>
<center>
<img src="docs/figure/teaser_v4.png">
</div>

## 🛠️ Setup

**Step 1:** Run the following commands step by step to set the environments:

```bash
conda create -n CHAIC python=3.9
conda activate CHAIC
pip3 install -e .
pip3 install torch==1.13.1+cu117 torchvision==0.14.1+cu117 \
 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
```

If you are running in a remote server without a screen, please refer to [running TDW in server](https://github.com/threedworld-mit/tdw/blob/master/Documentation/lessons/setup/server.md).

After that, you can run the demo scene to verify your setup:

```bash
python demo/demo_scene.py
```

**Step 2:** Install and download pre-trained perception models:

```bash
pip install -U openmim
mim install mmengine
mim install mmdet
pip install mmaction2
bash detection_pipeline/download_ckpt.sh
```

After that, you can run the perception demos to verify them:

```bash
python tdw-gym/detection.py
python tdw-gym/behavior.py
```

**Notice:** There may exist some internal bugs in the `mmaction` package, and you can refer to the [Github issue](https://github.com/open-mmlab/mmaction2/issues/2714) to fix it when you meet trouble.

## 💾 Codebase Layout
Some important folders and their corresponding functions are listed here.
```
|__ tdw-gym/                         Main code
|
|__ scenes/                          Code for dataset generation
|
|__ dataset/                         Dataset configuration and storage
|
|__ transport_challenge_multi_agent/ Low level controller
|
|__ scripts/                         Scripts for running experiments
|
|__ detection_pipeline/              Code for perception models
|
|__ LM_agent/                        LLM & VLM Prompt
```

## 💫 Run Experiments

We prepare all the experiment scripts to run experiments under the folder `scripts`. For example, to run experiments with Random Helper in the highthing setting, you can use the following command:

```bash
bash scripts/random_helper/test_high_thing_random_helper.sh
```

By adding ``--gt_mask`` or ``--gt_behavior`` in the scripts, the environment will provide ground truth object segmentation masks or ground truth behaviors of the partner, respectively.

**Notice:** If you want to test the LLM+BM helper or the VLM helper, you need to fill your ``AzureOpenAI`` setting or ``OPENAI_API_KEY`` at lines 73-77 in ``LM_agent/LLM.py`` or lines 74-78 in ``LM_agent/VLM.py``.

## 🧾 Benchmark Highlights

### Multi-Agent Asynchronized Setting

Agents may take different number of frames to finish (or fail) one action, and one env step is finished until any agent's action is not under the ongoing status, and the current obs is returned to all agents. Then, 
all agents are asked for a new action, and any agent having ongoing action will directly switch to the new action if its action changes. 

### Heterogeneity of Agents

Different types of agents have different capacity scopes, and agents with different capacity scopes need to work together to achieve a common goal. Meanwhile, although the task goal is the same for all agents, the constrained agent and the helper will receive different information about the goal: The constrained agent can know the exact goal of the task, while the helper needs to perceive the constrained agent's behavior and infer the true goal.

### Realistic Observation

One goal of our benchmark is to mimic real life as similar as possible. Therefore, we only provide the raw RGBD images as the main observation (the benchmark also supports many other types of observation), making our benchmark challenging and having a wide range of application space.

## 🤖 Creating a new agent

First you should learn about the details of the observation. The environment returns each agent's observation every step, which is a dictionary that includes the following items:

- **rgb**: RGB image of the current agent's view
- **depth**: depth image of the current agent's view
- **camera_matrix**: the camera matrix of current agent's ego camera
- **held_objects**: all the objects that current agent is holding. It is a list of length 2 that contains the information of the object that is held in the agent's two hands. Each object's information contains its name, type and a unique id. If it's a container, it also includes the information of the objects in it.
- **status**: the status of current action, which is a number from 0 to 2. 0 for 'ongoing', 1 for 'failure', 2 for 'success.
- **current_frames**: the number of frames passed
- **previous_action** & **previous_status**: all previous actions of the agent and their corresponding status
  
To create a new agent, you must first create a folder named 'agent' in the root directory of the repository, and create a python file in it to write your own agent. You need to implement the following two functions in the python file:

```python
def reset(obs, info):
def act(obs):
```

The function **reset** is used for initializing the agent at the beginning of the episode. It receives two arguments, that 'obs' is the initial observation of the agent, and 'info' is the information of the task. 

The function **act** is the core part of the agent. It determines the next action of the agent. It receives the current observation from the environment, and returns the action. Each action should be a dictionary and set its "type" key to an integer between 0 and 7, each refers to a certain type of action:

- 0: move forward by 0.5 meters
- 1: turn left by 15 degrees
- 2: turn right by 15 degrees
- 3: pick up an object, it should contain another key named 'object' whose value is the id of object to pick, together with a key named 'arm' representing which hand to pick. 0 for left hand, 1 for right hand.
- 4: put the object in one hand to the container in other hand. 
- 5: put the object on some surface, it should contain a key named 'object' whose value is the id of object to put on its surface.
- 6: remove obstacle, it should contain another key named 'object' whose value is the id of obstacle to pick, together with a key named 'arm' representing which hand to pick.
- 7: wait for several frames, it should contain a key named 'delay' indicating the number of frames to wait.

To evaluate your agent on a certain task, you should create a script like the following.

```bash
bash scripts/plan_helper/test_{your_task}_plan_helper.sh
```

You should change the second items of the 'agents' argument, which represents the type of the helper, to the name of the python file of your implemented agent. Then you can just run the script and get the result. You can also change the 'output_dir' of the script to customize the position to save the result.



## 🏆 Results

The table below is the quantitative results on CHAIC benchmark. We report the average Transport Rate (TR), Efficiency Improvement (EI), Goal Inference Accuracy (IA), Completion Ratio of Helper (CR) and Standard Error of Transport Rate (STD_TR) here. w/o means the main agent does the task solely without a helper. The Emergency Rate (ER) metric is also reported for the shopping task. 

<table>
    <tr>
        <td colspan="1"> <b>TR(EI)<span>&#8593;</span></b> </td>
        <td colspan="6" align="center">Indoor</td>
        <td colspan="2" align="center">Outdoor</td>
    </tr>
    <tr>
        <td>Helper Agent</td>
        <td>Normal</td>
        <td>High Target</td>
        <td>High Container</td>
        <td>High Goalplace</td>
        <td>Lowthing</td>
        <td>Wheelchair</td>
        <td>Shopping</td>
        <td>Furniture</td>
    </tr>
    <tr>
        <td>w/o</td>
        <td>0.53</td>
        <td>0.30</td>
        <td>0.37</td>
        <td>0.28</td>
        <td>0.51</td>
        <td>0.07</td>
        <td>0.37</td>
        <td>0.17</td>
    </tr>
    <tr>
        <td>Random</td>
        <td>0.52(-0.02)</td>
        <td>0.27(-0.05)</td>
        <td>0.36(0.00)</td>
        <td>0.33(0.10)</td>
        <td>0.50(-0.01)</td>
        <td>0.21(0.56)</td>
        <td>0.39(0.05)</td>
        <td>0.48(0.68)</td>
    </tr>
    <tr>
        <td>RHP</td>
        <td>0.64(0.15)</td>
        <td>0.35(0.11)</td>
        <td>0.45(0.19)</td>
        <td>0.35(0.18)</td>
        <td>0.66(0.23)</td>
        <td><b>0.44</b>(0.77)</td>
        <td>0.49(0.22)</td>
        <td>0.65(0.72)</td>
    </tr>
    <tr>
        <td>RL</td>
        <td>0.45(-0.19)</td>
        <td>0.26(-0.16)</td>
        <td>0.28(-0.25)</td>
        <td>0.25(-0.22)</td>
        <td>0.43(-0.16)</td>
        <td>0.11(0.07)</td>
        <td>0.32(-0.13)</td>
        <td>0.67(0.74)</td>
    </tr>
    <tr>
        <td>SmartHelp</td>
        <td>0.46(-0.12)</td>
        <td>0.24(-0.17)</td>
        <td>0.26(-0.28)</td>
        <td>0.31(0.01)</td>
        <td>0.49(-0.04)</td>
        <td>0.13(0.11)</td>
        <td>0.32(-0.13)</td>
        <td>0.57(0.70)</td>
    </tr>
    <tr>
        <td>VLM</td>
        <td>0.63(0.14)</td>
        <td>0.33(0.06)</td>
        <td>0.43(0.12)</td>
        <td>0.26(-0.20)</td>
        <td>0.69(0.26)</td>
        <td>0.40(0.86)</td>
        <td>0.50(0.25)</td>
        <td><b>0.70(0.78)</b></td>
    </tr>
    <tr>
        <td>LLM+BM</td>
        <td><b>0.65(0.17)</b></td>
        <td><b>0.38(0.19)</b></td>
        <td><b>0.49(0.24)</b></td>
        <td><b>0.36(0.23)</b></td>
        <td><b>0.70(0.27)</b></td>
        <td>0.42(<b>0.89</b>)</td>
        <td><b>0.58(0.33)</b></td>
        <td>0.69(0.77)</td>
    </tr>
    <tr>
        <td>Oracle</td>
        <td>0.77(0.31)</td>
        <td>0.49(0.37)</td>
        <td>0.69(0.47)</td>
        <td>0.61(0.56)</td>
        <td>0.82(0.38)</td>
        <td>0.60(0.87)</td>
        <td>0.61(0.39)</td>
        <td>0.76(0.80)</td>
    </tr>
</table>
<table>
    <tr>
        <td colspan="1"> <b>IA<span>&#8593;</span></b> </td>
        <td colspan="6" align="center">Indoor</td>
        <td colspan="1" align="center">Outdoor</td>
    </tr>
    <tr>
        <td>Helper Agent</td>
        <td>Normal</td>
        <td>High Target</td>
        <td>High Container</td>
        <td>High Goalplace</td>
        <td>Lowthing</td>
        <td>Wheelchair</td>
        <td>Shopping</td>
    </tr>
    <tr>
        <td>Random</td>
        <td>0.24</td>
        <td>0.29</td>
        <td>0.25</td>
        <td>0.14</td>
        <td>0.31</td>
        <td>0.24</td>
        <td>0.34</td>
    </tr>
    <tr>
        <td>RHP</td>
        <td>0.15</td>
        <td>0.29</td>
        <td>0.21</td>
        <td>0.21</td>
        <td>0.28</td>
        <td>0.17</td>
        <td>0.44</td>
    </tr>
    <tr>
        <td>VLM</td>
        <td>0.24</td>
        <td><b>0.32</b></td>
        <td><b>0.40</b></td>
        <td>0.33</td>
        <td><b>0.46</b></td>
        <td>0.35</td>
        <td>0.72</td>
    </tr>
    <tr>
        <td>LLM+BM</td>
        <td><b>0.25</b></td>
        <td>0.29</td>
        <td>0.30</td>
        <td><b>0.35</b></td>
        <td>0.43</td>
        <td><b>0.47</b></td>
        <td><b>0.74</b></td>
    </tr>
    <tr>
        <td>Oracle</td>
        <td>0.88</td>
        <td>0.91</td>
        <td>0.91</td>
        <td>0.90</td>
        <td>0.91</td>
        <td>0.82</td>
        <td>0.87</td>
    </tr>
</table>
<table>
    <tr>
        <td colspan="1"> <b>CR<span>&#8593;</span></b> </td>
        <td colspan="6" align="center">Indoor</td>
        <td colspan="2" align="center">Outdoor</td>
    </tr>
    <tr>
        <td>Helper Agent</td>
        <td>Normal</td>
        <td>High Target</td>
        <td>High Container</td>
        <td>High Goalplace</td>
        <td>Lowthing</td>
        <td>Wheelchair</td>
        <td>Shopping</td>
        <td>Furniture</td>
    </tr>
    <tr>
        <td>Random</td>
        <td>0.09</td>
        <td>0.10</td>
        <td>0.12</td>
        <td>0.06</td>
        <td>0.09</td>
        <td>0.09</td>
        <td>0.07</td>
        <td>0.73</td>
    </tr>
    <tr>
        <td>RHP</td>
        <td>0.15</td>
        <td><b>0.43</b></td>
        <td>0.29</td>
        <td><b>0.39</b></td>
        <td>0.36</td>
        <td>0.19</td>
        <td>0.34</td>
        <td>0.74</td>
    </tr>
    <tr>
        <td>VLM</td>
        <td>0.13</td>
        <td>0.08</td>
        <td><b>0.34</b></td>
        <td>0.18</td>
        <td><b>0.39</b></td>
        <td>0.17</td>
        <td>0.34</td>
        <td><b>0.82</b></td>
    </tr>
    <tr>
        <td>LLM+BM</td>
        <td><b>0.22</b></td>
        <td>0.30</td>
        <td>0.30</td>
        <td>0.35</td>
        <td>0.38</td>
        <td><b>0.45</b></td>
        <td><b>0.46</b></td>
        <td>0.78</td>
    </tr>
    <tr>
        <td>Oracle</td>
        <td>0.51</td>
        <td>0.64</td>
        <td>0.66</td>
        <td>0.73</td>
        <td>0.59</td>
        <td>0.38</td>
        <td>0.45</td>
        <td>0.77</td>
    </tr>
</table>
<table>
    <tr>
        <td colspan="1"> <b>STD</b></td>
        <td colspan="6" align="center">Indoor</td>
        <td colspan="2" align="center">Outdoor</td>
    </tr>
    <tr>
        <td>Helper Agent</td>
        <td>Normal</td>
        <td>High Target</td>
        <td>High Container</td>
        <td>High Goalplace</td>
        <td>Lowthing</td>
        <td>Wheelchair</td>
        <td>Shopping</td>
        <td>Furniture</td>
    </tr>
    <tr>
        <td>w/o</td>
        <td>0.03</td>
        <td>0.02</td>
        <td>0.03</td>
        <td>0.05</td>
        <td>0.03</td>
        <td>0.04</td>
        <td>0.02</td>
        <td>0.04</td>
    </tr>
    <tr>
        <td>Random</td>
        <td>0.04</td>
        <td>0.03</td>
        <td>0.03</td>
        <td>0.04</td>
        <td>0.04</td>
        <td>0.04</td>
        <td>0.02</td>
        <td>0.05</td>
    </tr>
    <tr>
        <td>RHP</td>
        <td>0.02</td>
        <td>0.04</td>
        <td>0.03</td>
        <td>0.05</td>
        <td>0.03</td>
        <td>0.04</td>
        <td>0.02</td>
        <td>0.04</td>
    </tr>
    <tr>
        <td>VLM</td>
        <td>0.03</td>
        <td>0.02</td>
        <td>0.04</td>
        <td>0.05</td>
        <td>0.02</td>
        <td>0.03</td>
        <td>0.03</td>
        <td>0.05</td>
    </tr>
    <tr>
        <td>LLM+BM</td>
        <td>0.03</td>
        <td>0.03</td>
        <td>0.03</td>
        <td>0.04</td>
        <td>0.03</td>
        <td>0.05</td>
        <td>0.03</td>
        <td>0.05</td>
    </tr>
    <tr>
        <td>Oracle</td>
        <td>0.03</td>
        <td>0.04</td>
        <td>0.03</td>
        <td>0.04</td>
        <td>0.03</td>
        <td>0.03</td>
        <td>0.03</td>
        <td>0.04</td>
    </tr>
</table>
<table>
    <tr>
        <td colspan="1"> <b>ER<span>&#8595;</span></b> </td>
        <td colspan="1" align="center">Outdoor</td>
    </tr>
    <tr>
        <td>Helper Agent</td>
        <td>Shopping</td>
    </tr>
    <tr>
        <td>Random</td>
        <td>0.32</td>
    </tr>
    <tr>
        <td>RHP</td>
        <td><b>0.30</b></td>
    </tr>
    <tr>
        <td>VLM</td>
        <td>0.39</td>
    </tr>
    <tr>
        <td>LLM+BM</td>
        <td>0.38</td>
    </tr>
    <tr>
        <td>Oracle</td>
        <td>0.17</td>
    </tr>
</table>
