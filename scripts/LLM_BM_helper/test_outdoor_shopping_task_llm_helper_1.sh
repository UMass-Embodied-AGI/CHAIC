port=11000
pkill -f -9 "port $port"

python tdw-gym/challenge.py \
--output_dir results \
--experiment_name outdoor_shopping_task_llm_helper \
--run_id test \
--port $port \
--agents plan_agent plan_agent child_agent \
--plan_mode default LLM default \
--max_tokens 512 \
--lm_id gpt-4 \
--max_frames 3000 \
--data_prefix dataset/test_dataset/outdoor_shopping \
--prompt_template_path "./LM_agent/modified_prompts/prompt_helper_bike.csv" \
--screen_size 512 \
--debug \
--cot \
--only_save_rgb

ps ux | grep port\ $port | awk {'print $2'} | xargs kill