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
  for m in `ls`
  do
    if [ -d "$m" ]
    then
      mkv="$m"_down4.mkv
      decode_dir="$curdir"/sequences_qp37_down4/"$d"/"$m"
      mkdir -p $decode_dir
      echo "decoding $mkv into $decode_dir"
      echo ""
      echo ""
      $HOME/ffmpeg-4.3.1-amd64-static/ffmpeg -i $mkv -start_number 0 $decode_dir/im%d.png

      mkv="$m"_down2.mkv
      decode_dir="$curdir"/sequences_qp37_down2/"$d"/"$m"
      mkdir -p $decode_dir
      echo "decoding $mkv into $decode_dir"
      echo ""
      echo ""
      $HOME/ffmpeg-4.3.1-amd64-static/ffmpeg -i $mkv -start_number 0 $decode_dir/im%d.png

      mkv="$m".mkv
      decode_dir="$curdir"/sequences_qp37/"$d"/"$m"
      mkdir -p $decode_dir
      echo "decoding $mkv into $decode_dir"
      echo ""
      echo ""
      $HOME/ffmpeg-4.3.1-amd64-static/ffmpeg -i $mkv -start_number 0 $decode_dir/im%d.png
    fi
  done
  cd $curdir/sequences
done

