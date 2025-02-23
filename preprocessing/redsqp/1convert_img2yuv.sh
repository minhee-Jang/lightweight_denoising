#!/bin/bash


curdir=`pwd`
echo $curdir

echo "change directory to 'hr'"
cd 'hr'

for d in `ls`
do
  echo "encoding $d into $d.mkv"
  if [ -d "$d" ]
  then
    $HOME/ffmpeg-4.3.1-amd64-static/ffmpeg -r 30 -start_number 0 -i "$d/%08d.png" -f rawvideo -pix_fmt yuv420p "$d.yuv"
  fi
done

