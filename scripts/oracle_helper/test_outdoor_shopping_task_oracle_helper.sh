port=11000
pkill -f -9 "port $port"

python tdw-gym/challenge.py \
--output_dir results \
--experiment_name outdoor_shopping_task_oracle_helper \
--run_id test \
--port $port \
--agents plan_agent plan_agent child_agent \
--plan_mode default default default \
--prompt_template_path LLM/prompt_nocom.csv \
--max_tokens 256 \
--cot \
--lm_id gpt-4 \
--max_frames 3000 \
--data_prefix dataset/test_dataset/outdoor_shopping \
--screen_size 256 \
--oracle

ps ux | grep port\ $port | awk {'print $2'} | xargs kill