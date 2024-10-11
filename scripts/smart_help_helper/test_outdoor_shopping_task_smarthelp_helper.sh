port=11000
pkill -f -9 "port $port"

python tdw-gym/challenge.py \
--output_dir results \
--experiment_name outdoor_shopping_task_smarthelp_helper \
--run_id test \
--port $port \
--agents plan_agent plan_agent child_agent \
--plan_mode default smart_help default \
--prompt_template_path LLM/prompt_nocom.csv \
--max_tokens 256 \
--cot \
--max_frames 3000 \
--lm_id gpt-4 \
--data_prefix dataset/test_dataset/outdoor_shopping \
--screen_size 512 \
--debug \
--eval_episodes 0 1 2 3 4 5 6 7 8 9 10 11 \
--gt_mask

ps ux | grep port\ $port | awk {'print $2'} | xargs kill