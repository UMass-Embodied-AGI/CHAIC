import sys
import os
import subprocess
import argparse
import glob
import re

def create_video(video_folder, out_file):
    frame_rate = 2

    # 使用glob来获取文件名列表
    files = sorted(glob.glob(os.path.join(video_folder, "*_*.png")))

    # 使用临时文件名列表文件
    filelist = os.path.join(video_folder, "filelist.txt")
    with open(filelist, "w") as f:
        for file in files:
            if re.match(r"\d+_\d+\.png", os.path.basename(file)):
                f.write(f"file '{file}'\n")

    # 使用绝对路径指向ffmpeg可执行文件
    ffmpeg_executable = './ffmpeg-6.0-amd64-static/ffmpeg'
    
    if not os.path.exists(ffmpeg_executable):
        print(f"Error: ffmpeg executable not found at {ffmpeg_executable}")
        sys.exit(1)

    # 调用subprocess来运行ffmpeg命令
    subprocess.call([ffmpeg_executable,
                     '-r', str(frame_rate),
                     '-f', 'concat',
                     '-safe', '0',
                     '-i', filelist,
                     '-pix_fmt', 'yuv420p',
                     '-r', str(frame_rate),
                     out_file])

    # 删除临时文件名列表文件
    os.remove(filelist)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default='Images', type=str)
    parser.add_argument("--output", default='demo.mp4', type=str)
    args = parser.parse_args()

    create_video(args.dir, args.output)
