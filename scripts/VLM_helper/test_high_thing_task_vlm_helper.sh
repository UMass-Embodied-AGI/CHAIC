port=18098
pkill -f -9 "port $port"

python tdw-gym/challenge.py \
--output_dir results \
--experiment_name highthing_task_vlm_helper \
--run_id test \
--port $port \
--agents plan_agent plan_agent \
--plan_mode default VLM \
--vlm_prompt_template_path LM_agent/modified_prompts/prompt_helper_highthing.csv \
--max_tokens 512 \
--vlm_id gpt-4o \
--max_frames 3000 \
--data_prefix dataset/test_dataset/highthing \
--screen_size 512 \
--gt_behavior \
--debug

ps ux | grep port\ $port | awk {'print $2'} | xargs kill