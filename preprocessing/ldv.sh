#!/bin/bash

dir_list="
train_down4_QP37
train_down4_gt
validation_track3
"

curdir=`pwd`
echo $curdir

for d in $dir_list
do
  echo "change directory to $d"
  cd $d

  for f in `ls`
  do
      #echo $f
      #fn=$f
      dn=${f:0:3}
      echo "making directory $dn"
      mkdir $dn
      ffmpeg -i $f $dn/f%3d.png
  done

  cd $curdir
done