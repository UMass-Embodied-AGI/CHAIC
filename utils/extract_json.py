import os
import shutil

def copy_json_files(src_dir, dst_dir):
    os.makedirs(dst_dir, exist_ok=True)
    
    for root, dirs, files in os.walk(src_dir):
        for file in files:
            if file.endswith('.json') and "chat" not in file:
                src_file_path = os.path.join(root, file)
                dst_file_path = os.path.join(dst_dir, os.path.relpath(src_file_path, start=src_dir))
                
                os.makedirs(os.path.dirname(dst_file_path), exist_ok=True)
                shutil.copy2(src_file_path, dst_file_path)
                print(f"Copied: {src_file_path} to {dst_file_path}")

# 使用示例
source_directory = "organized_result/"
destination_directory = "organized_result_only_json/"
copy_json_files(source_directory, destination_directory)
