#!/bin/bash


if [ -z "$1" ]
then
  START_IDX=1
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
    for y in `ls *.yuv | grep -E '^[0-9]+\.yuv$'`
    do
        echo "downscaling $y"
        yn=${y:0:4}
        $HOME/ffmpeg-4.3.1-amd64-static/ffmpeg -pix_fmt yuv420p -s 448x256 -i $y -vf scale=224x128:flags=bicubic "$yn"_down2.yuv
        $HOME/ffmpeg-4.3.1-amd64-static/ffmpeg -pix_fmt yuv420p -s 448x256 -i $y -vf scale=112x64:flags=bicubic "$yn"_down4.yuv
    done

    cd $curdir/sequences
done

