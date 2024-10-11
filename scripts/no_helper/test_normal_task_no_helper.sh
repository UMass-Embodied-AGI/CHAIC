port=14001
pkill -f -9 "port $port"

python tdw-gym/challenge.py \
--output_dir results \
--experiment_name normal_task_no_helper \
--run_id test \
--port $port \
--agents plan_agent \
--plan_mode default \
--prompt_template_path kfcvw50 \
--max_tokens 256 \
--cot \
--lm_id gpt-4 \
--max_frames 3000 \
--data_prefix dataset/test_dataset/normal \
--screen_size 512 \
--seed_num 1 \
--debug

ps ux | grep port\ $port | awk {'print $2'} | xargs kill