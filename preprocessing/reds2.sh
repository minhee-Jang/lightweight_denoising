#!/bin/bash
# I cannot run scale 4 due to the CU size issue.

curdir=`pwd`
echo $curdir

echo "change directory to 'hr'"
cd 'hr'

for y in `ls | grep -E '^[0-9]+\.yuv$'`
do
    echo "downscaling $y"
    yn=${y:0:3}
    $HOME/ffmpeg-4.3.1-amd64-static/ffmpeg -pix_fmt yuv420p -s 1280x720 -i $y -vf scale=640x360:flags=bicubic "$yn"_down2.yuv
done

for y in `ls *_down2.yuv`
do
  echo "compressing $y"
  yn=${y:0:3}
  $HOME/HM16.20/bin/TAppEncoderStatic \
    -c $HOME/HM16.20/cfg/encoder_lowdelay_P_main.cfg \
    -c $HOME/HM16.20/cfg/per-sequence/BasketballDrill.cfg \
    -i $y -q 37 -wdt 640 -hgt 360 -f 100 -fr 30  \
    -b "$yn_down2.mkv"
done

for m in `ls *_down2.mkv`
do
  echo "decoding $m"
  dn=${m:0:3}_down2
  echo "making directory $dn"
  mkdir $curdir/lr/$dn
  $HOME/ffmpeg-4.3.1-amd64-static/ffmpeg -s 640x360 -i $m -start_number 0 $curdir/lr/$dn/%08d.png
done