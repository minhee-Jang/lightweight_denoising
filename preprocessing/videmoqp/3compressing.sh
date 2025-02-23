#!/bin/bash

if [ $# -lt 1 ]
then
  echo "please run $0 <start idx> <end idx>"
  exit 1
fi

if [ -z "$1" ]
then
  START_IDX=0
else
  START_IDX=$1
fi

if [ -z "$2" ]
then
  END_IDX=96
else
  END_IDX=$2
fi

echo "Range($START_IDX:$END_IDX)"

curdir=`pwd`
echo $curdir

echo "change directory to 'sequences'"
cd 'sequences'

for d in $(seq -f "%05g" $START_IDX $END_IDX )
do
  echo "change directory into $d"
  cd $d
  for m in `ls`
  do
    if [ -d "$m" ]
    then
      yuv="$m"_down4.yuv
      mkv="$m"_down4.mkv
      echo "compressing $yuv into $mkv"
      echo ""
      echo ""
      $HOME/HM16.20/bin/TAppEncoderStatic \
          -c $HOME/HM16.20/cfg/encoder_lowdelay_P_main.cfg \
          -c $HOME/HM16.20/cfg/per-sequence/BasketballDrill.cfg \
          -i $yuv -q 37 -wdt 112 -hgt 64 -f 7 -fr 30  \
          -b $mkv
          
      yuv="$m"_down2.yuv
      mkv="$m"_down2.mkv
      echo "compressing $yuv into $mkv"
      echo ""
      echo ""
      $HOME/HM16.20/bin/TAppEncoderStatic \
          -c $HOME/HM16.20/cfg/encoder_lowdelay_P_main.cfg \
          -c $HOME/HM16.20/cfg/per-sequence/BasketballDrill.cfg \
          -i $yuv -q 37 -wdt 224 -hgt 128 -f 7 -fr 30  \
          -b $mkv

      yuv=$m.yuv
      mkv=$m.mkv
      echo "compressing $yuv into $mkv"
      echo ""
      echo ""
      $HOME/HM16.20/bin/TAppEncoderStatic \
          -c $HOME/HM16.20/cfg/encoder_lowdelay_P_main.cfg \
          -c $HOME/HM16.20/cfg/per-sequence/BasketballDrill.cfg \
          -i $yuv -q 37 -wdt 448 -hgt 256 -f 7 -fr 30  \
          -b $mkv
    fi
  done
  cd $curdir/sequences
done

