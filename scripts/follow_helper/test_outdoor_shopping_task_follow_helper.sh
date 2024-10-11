port=18001
pkill -f -9 "port $port"

python tdw-gym/challenge.py \
--output_dir results \
--experiment_name outdoor_shopping_task_follow_helper \
--run_id train_data_collection \
--port $port \
--agents plan_agent follow_agent \
--plan_mode default default \
--prompt_template_path LLM/prompt_nocom.csv \
--max_tokens 256 \
--lm_id gpt-4 \
--max_frames 3000 \
--data_prefix dataset/train_dataset/outdoor_shopping \
--screen_size 512 \
--gt_mask \
--debug

ps ux | grep port\ $port | awk {'print $2'} | xargs kill