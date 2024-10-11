port=11005
pkill -f -9 "port $port"

python tdw-gym/challenge.py \
--output_dir results \
--experiment_name high_container_task_plan_helper \
--run_id test \
--port $port \
--agents plan_agent plan_agent \
--plan_mode default default \
--prompt_template_path LLM/prompt_nocom.csv \
--max_tokens 256 \
--cot \
--lm_id gpt-4 \
--max_frames 3000 \
--data_prefix dataset/test_dataset/highcontainer \
--screen_size 512

ps ux | grep port\ $port | awk {'print $2'} | xargs kill