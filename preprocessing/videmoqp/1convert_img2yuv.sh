#!/bin/bash
# Remove in train list!!!
# train
# 00005/0379
# 00005/0513
# 00013/0178
# 00015/0462
# 00016/0464
# 00016/0465
# 00016/0466
# 00017/0102
# 00022/0934
# 00022/0937
# 00027/0585
# 00027/0586
# 00027/0587
# 00030/0556
# 00030/0557
# 00030/0558
# 00030/0559
# 00032/0845
# 00032/0846
# 00038/0362
# 00038/0363
# 00038/0364
# 00038/0365
# 00038/0511
# 00038/0512
# 00042/0954
# 00042/0955
# 00044/0111
# 00046/0984
# 00047/0416
# 00047/0417
# 00053/0823
# 00066/0232
# 00066/0981
# 00066/0982
# 00072/0654
# 00073/0395
# 00073/0396
# 00073/0397
# 00073/0400
# 00073/0669
# 00073/0670
# 00073/0671
# 00073/0672
# 00073/0880
# 00073/0881
# 00073/0882
# 00080/0366
# 00081/0294
# 00095/0031
# 00096/0290
# 00096/0291
# test
# 00003/0531
# 00004/0569
# 00049/0900
# 00066/0694
# 00077/0480

# Valid
# 00003/0532
# 00004/0570
# 00004/0571

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
            echo ""
            echo "encoding $m into $m.yuv"
            $HOME/ffmpeg-4.3.1-amd64-static/ffmpeg -r 30 -y -i "$m/im%d.png" -f rawvideo -pix_fmt yuv420p "$m.yuv"
        fi
    done

    cd $curdir/sequences
done

