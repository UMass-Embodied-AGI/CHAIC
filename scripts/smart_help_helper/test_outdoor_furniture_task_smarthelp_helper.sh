port=11000
pkill -f -9 "port $port"

python tdw-gym/challenge.py \
--output_dir results \
--experiment_name outdoor_furniture_task_smarthelp_helper \
--run_id test \
--port $port \
--agents plan_agent plan_agent \
--plan_mode default smart_help \
--prompt_template_path LLM/prompt_nocom.csv \
--max_tokens 256 \
--cot \
--max_frames 3000 \
--lm_id gpt-4 \
--data_prefix dataset/test_dataset/outdoor_furniture \
--screen_size 512 \
--debug \
--gt_mask

ps ux | grep port\ $port | awk {'print $2'} | xargs kill