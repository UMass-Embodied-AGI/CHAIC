ffmpeg -framerate 30 -i $1/img_%05d.jpg -c:v libx264 -r 30 -pix_fmt yuv420p $2.mp4
ffmpeg -framerate 30 -i $1/%04d.png -c:v libx264 -r 30 -pix_fmt yuv420p $2.mp4