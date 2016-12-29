#!/usr/bin/env bash

# Inter-subjet recognition of 8 gestures in CapgMyo DB-b
for i in $(seq 0 9 | shuf); do
  scripts/rundocker.sh python -m sigr.app exp --log log --snapshot model \
    --num-epoch 28 --lr-step 16 --lr-step 24 --snapshot-period 28 \
    --root .cache/sensors-dbb-inter-subject-$i \
    --num-semg-row 16 --num-semg-col 8 \
    --batch-size 1000 --decay-all --dataset dbb \
    --num-filter 64 \
    --adabn --minibatch \
    crossval --crossval-type inter-subject --fold $i
done

# Inter-session recognition of 8 gestures in CapgMyo DB-b
for i in 1; do
  scripts/rundocker.sh python -m sigr.app exp --log log --snapshot model \
    --num-epoch 28 --lr-step 16 --lr-step 24 --snapshot-period 28 \
    --root .cache/sensors-dbb-universal-inter-session-$i \
    --num-semg-row 16 --num-semg-col 8 \
    --batch-size 1000 --decay-all --dataset dbb \
    --num-filter 64 \
    --adabn --minibatch \
    crossval --crossval-type universal-inter-session --fold $i
done
for i in $(seq 1 2 19 | shuf); do
  scripts/rundocker.sh python -m sigr.app exp --log log --snapshot model \
    --num-epoch 28 --lr-step 16 --lr-step 24 --snapshot-period 28 \
    --root .cache/sensors-dbb-inter-session-$i \
    --num-semg-row 16 --num-semg-col 8 \
    --batch-size 1000 --decay-all --dataset dbb \
    --num-filter 64 \
    --params .cache/sensors-dbb-universal-inter-session-1/model-0028.params \
    --fix-params ".*conv.*" --fix-params ".*pixel.*" --fix-params "fc1_.*" --fix-params "fc2_.*" \
    --adabn \
    crossval --crossval-type inter-session --fold $i
done

# Inter-subjet recognition of 12 gestures in CapgMyo DB-c
for i in $(seq 0 9 | shuf); do
  scripts/rundocker.sh python -m sigr.app exp --log log --snapshot model \
    --num-epoch 28 --lr-step 16 --lr-step 24 --snapshot-period 28 \
    --root .cache/sensors-dbc-inter-subject-$i \
    --num-semg-row 16 --num-semg-col 8 \
    --batch-size 1000 --decay-all --dataset dbc \
    --num-filter 64 \
    --adabn --minibatch \
    crossval --crossval-type inter-subject --fold $i
done

# Inter-session recognition of 27 gestures in CSL-HDEMG
for i in $(seq 0 5 | shuf); do
  scripts/rundocker.sh python -m sigr.app exp --log log --snapshot model \
    --num-epoch 28 --lr-step 16 --lr-step 24 --snapshot-period 28 \
    --root .cache/sensors-csl-universal-inter-session-$i \
    --num-semg-row 24 --num-semg-col 7 \
    --batch-size 1000 --decay-all --adabn --minibatch --dataset csl \
    --preprocess '(csl-bandpass,csl-cut,downsample-5,median)' \
    --balance-gesture 1 \
    --num-filter 64 \
    crossval --crossval-type universal-inter-session --fold $i
done
for i in $(seq 0 24 | shuf); do
  scripts/rundocker.sh python -m sigr.app exp --log log --snapshot model \
    --num-epoch 28 --lr-step 16 --lr-step 24 --snapshot-period 28 \
    --root .cache/sensors-csl-inter-session-$i \
    --num-semg-row 24 --num-semg-col 7 \
    --batch-size 1000 --decay-all --adabn --minibatch --dataset csl \
    --preprocess '(csl-bandpass,csl-cut,median)' \
    --balance-gesture 1 \
    --num-filter 64 \
    --params .cache/sensors-csl-universal-inter-session-$(($i % 5))/model-0028.params \
    crossval --crossval-type inter-session --fold $i
done

# Inter-subject recognition of 52 gestures in NinaPro DB1 with calibration data
for i in 0; do
  scripts/rundocker.sh python -m sigr.app exp --log log --snapshot model \
    --num-epoch 28 --lr-step 16 --lr-step 24 --snapshot-period 28 \
    --root .cache/sensors-ninapro-universal-one-fold-intra-subject-$i \
    --num-semg-row 1 --num-semg-col 10 \
    --batch-size 1000 --decay-all --adabn --minibatch --dataset ninapro-db1/caputo \
    --num-filter 64 \
    --preprocess 'downsample-16' \
    crossval --crossval-type universal-one-fold-intra-subject --fold $i
done
for i in $(seq 0 26 | shuf); do
  scripts/rundocker.sh python -m sigr.app exp --log log --snapshot model \
    --num-epoch 28 --lr-step 16 --lr-step 24 --snapshot-period 28 \
    --root .cache/sensors-ninapro-one-fold-intra-subject-$i \
    --num-semg-row 1 --num-semg-col 10 \
    --batch-size 1000 --decay-all --dataset ninapro-db1/caputo \
    --num-filter 64 \
    --params .cache/sensors-ninapro-universal-one-fold-intra-subject-0/model-0028.params \
    --preprocess 'downsample-16' \
    crossval --crossval-type one-fold-intra-subject --fold $i
done
