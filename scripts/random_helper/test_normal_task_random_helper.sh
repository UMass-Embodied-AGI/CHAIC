port=12000
pkill -f -9 "port $port"

python tdw-gym/challenge.py \
--output_dir results \
--experiment_name normal_task_random_helper \
--run_id test \
--port $port \
--agents plan_agent plan_agent \
--plan_mode default random \
--prompt_template_path LLM/prompt_nocom.csv \
--max_tokens 256 \
--cot \
--lm_id gpt-4 \
--max_frames 3000 \
--data_prefix dataset/test_dataset/normal \
--screen_size 512 \
--no_save_img \
--seed_num 2 \
--debug

ps ux | grep port\ $port | awk {'print $2'} | xargs kill