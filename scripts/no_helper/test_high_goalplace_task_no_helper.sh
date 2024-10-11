port=11099
pkill -f -9 "port $port"

python tdw-gym/challenge.py \
--output_dir results \
--experiment_name high_goalplace_task_no_helper \
--run_id test \
--port $port \
--agents plan_agent \
--plan_mode default \
--prompt_template_path LLM/prompt_nocom.csv \
--max_tokens 256 \
--cot \
--lm_id gpt-4 \
--max_frames 3000 \
--data_prefix dataset/test_dataset/highgoalplace \
--screen_size 512 \
--debug

ps ux | grep port\ 11099 | awk {'print $2'} | xargs kill