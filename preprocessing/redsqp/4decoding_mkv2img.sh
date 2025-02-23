#!/bin/bash

#if ! [ $# -lt 1 ]
#then
#  echo "please run $0 <start index> <end index>"
#  exit 1
#fi


if [ "$1" -eq 2 ]
then
  SCALE_STR=_down2
  SCALE=2
  W=$(expr 1280 / $SCALE)
  H=$(expr 720 / $SCALE)
elif [ "$1" -eq 4 ]
then
  SCALE_STR=_down4
  SCALE=4
  W=$(expr 1280 / $SCALE)
  H=$(expr 720 / $SCALE)
else
  SCALE_STR=''
  SCALE=1
fi

echo "SCALE: $SCALE"
echo "$s"


curdir=`pwd`
echo $curdir

echo "change directory to 'hr'"
cd 'hr'


for m in `ls *.mkv | grep -E '^[0-9]+\'$SCALE_STR'.mkv$'`
do
  echo "decoding $m into lr$SCALE_STR"
  dn=${m:0:3}
  lrdir=$curdir/lr$SCALE_STR/$dn
  echo "making directory $lrdir"
  mkdir $lrdir

  $HOME/ffmpeg-4.3.1-amd64-static/ffmpeg -i $m -start_number 0 $lrdir/%08d.png
  echo ""
  echo ""
  echo ""
  echo ""
  echo ""
done