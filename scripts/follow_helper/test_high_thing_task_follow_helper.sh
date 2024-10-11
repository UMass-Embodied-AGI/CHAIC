port=18005
pkill -f -9 "port $port"

python tdw-gym/challenge.py \
--output_dir results \
--experiment_name high_thing_task_follow_helper \
--run_id behavior_train \
--port $port \
--agents plan_agent follow_agent \
--plan_mode default default \
--prompt_template_path LLM/prompt_nocom.csv \
--max_tokens 256 \
--lm_id gpt-4 \
--max_frames 3000 \
--data_prefix dataset/train_dataset/highthing \
--screen_size 512 \
--gt_mask \
--only_save_rgb

ps ux | grep port\ $port | awk {'print $2'} | xargs kill