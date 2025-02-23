#!/bin/bash
if [ $# -lt 1 ]
then
  echo "please run $0 <scale[2|4]> <index of being compressed [0|1|2]> <start idx> <end idx>"
  exit 1
fi

if [ $1 -eq 2 ]
then
  SCALE_STR=_down2
  SCALE=2
elif [ $1 -eq 4 ]
then
  SCALE_STR=_down4
  SCALE=4
  cu="--ConformanceWindowMode 1"
else
  SCALE_STR=''
  SCALE=1
fi

if [ -z "$2" ]
then
  START_IDX=0
else
  START_IDX=$2
fi

if [ -z "$3" ]
then
  END_IDX=239
else
  END_IDX=$3
fi

echo "Range($START_IDX:$END_IDX)"
echo "SCALE: $SCALE"

W=$(expr 1280 / $SCALE)
H=$(expr 720 / $SCALE)

echo "$W x $H"

curdir=`pwd`
echo $curdir

echo "change directory to 'hr'"
cd 'hr'



for y in $(seq -f "%03g" $START_IDX $END_IDX )
do
  yuv=$y$SCALE_STR.yuv
  mkv=$y$SCALE_STR.mkv
  echo "compressing $yuv into $mkv"
  echo ""
  echo ""
  $HOME/HM16.20/bin/TAppEncoderStatic \
    -c $HOME/HM16.20/cfg/encoder_lowdelay_P_main.cfg \
    -c $HOME/HM16.20/cfg/per-sequence/BasketballDrill.cfg \
    -i $yuv $cu -q 37 -wdt "$W" -hgt "$H" -f 100 -fr 30  \
    -b $mkv
  echo ""
  echo ""
  echo ""
done