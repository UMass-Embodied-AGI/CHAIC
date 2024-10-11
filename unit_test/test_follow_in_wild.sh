port=11005
pkill -f -9 "port $port"

python tdw-gym/challenge.py \
--output_dir results \
--experiment_name follow_in_room_4 \
--run_id test \
--port $port \
--agents plan_agent plan_agent \
--plan_mode default follow \
--prompt_template_path LLM/prompt_nocom.csv \
--max_tokens 256 \
--cot \
--lm_id gpt-4 \
--max_frames 3000 \
--data_prefix dataset/test_dataset/normal \
--screen_size 512 \
--gt_mask \
--debug

ps ux | grep port\ $port | awk {'print $2'} | xargs kill