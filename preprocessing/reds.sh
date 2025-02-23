#!/bin/bash

# make hr and lr directories.
# Save hr images in hr directory and run follwoing.

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

for y in `ls *.yuv`
do
  echo "compressing $y"
  yn=${y:0:3}
  $HOME/HM16.20/bin/TAppEncoderStatic \
    -c $HOME/HM16.20/cfg/encoder_lowdelay_P_main.cfg \
    -c $HOME/HM16.20/cfg/per-sequence/BasketballDrill.cfg \
    -i $y -q 37 -wdt 1280 -hgt 720 -f 100 -fr 30  \
    -b "$yn.mkv"
done

for m in `ls *.mkv`
do
  echo "decoding $m"
  dn=${m:0:3}
  echo "making directory $dn"
  mkdir $curdir/lr/$dn
  $HOME/ffmpeg-4.3.1-amd64-static/ffmpeg -s 1280x720 -i $m -start_number 0 $curdir/lr/$dn/%08d.png
done