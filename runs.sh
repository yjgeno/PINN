#!/bin/bash

trap break INT
for i in `seq 10 $max`
do
  echo "run: $i"
  python -m Burgers.train --log_dir logdir/train_$i -n 500 -nu 0.01 -Ns 10000 -l -N $i
  echo "run $i completed"
done
trap - INT
