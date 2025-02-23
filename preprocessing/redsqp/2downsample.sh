#!/bin/bash


curdir=`pwd`
echo $curdir

echo "change directory to 'hr'"
cd 'hr'

for y in `ls *.yuv | grep -E '^[0-9]+\.yuv$'`
do
  echo "downscaling $y"
  yn=${y:0:3}
  $HOME/ffmpeg-4.3.1-amd64-static/ffmpeg -pix_fmt yuv420p -s 1280x720 -i $y -vf scale=640x360:flags=bicubic "$yn"_down2.yuv
  $HOME/ffmpeg-4.3.1-amd64-static/ffmpeg -pix_fmt yuv420p -s 1280x720 -i $y -vf scale=320x180:flags=bicubic "$yn"_down4.yuv
done
