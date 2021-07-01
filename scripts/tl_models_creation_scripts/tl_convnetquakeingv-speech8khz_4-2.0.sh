#!/bin/bash

python train.py convnetquakeingv speech8khz1k5_4 classification --epochs 250 --lr 0.001 --lr-reducer --early-stopping --name convnetquakeingv-lomax-speech8khz1k5_4-$LR_MULTIPLIER --lr-multiplier $LR_MULTIPLIER --weights convnetquakeingv-lomax/ --use-max-stream --cudnn-lstm --save-best --save-all --log-to-file --verbose 2
python train.py convnetquakeingv speech8khz9k_4 classification --epochs 250 --lr 0.001 --lr-reducer --early-stopping --name convnetquakeingv-lomax-speech8khz9k_4-$LR_MULTIPLIER --lr-multiplier $LR_MULTIPLIER --weights convnetquakeingv-lomax/ --use-max-stream --cudnn-lstm --save-best --save-all --log-to-file --verbose 2
python train.py convnetquakeingv speech8khz1k5_4 classification --epochs 250 --lr 0.001 --lr-reducer --early-stopping --name convnetquakeingv-lendb-speech8khz1k5_4-$LR_MULTIPLIER --lr-multiplier $LR_MULTIPLIER --weights convnetquakeingv-lendb/ --use-max-stream --cudnn-lstm --save-best --save-all --log-to-file --verbose 2
python train.py convnetquakeingv speech8khz9k_4 classification --epochs 250 --lr 0.001 --lr-reducer --early-stopping --name convnetquakeingv-lendb-speech8khz9k_4-$LR_MULTIPLIER --lr-multiplier $LR_MULTIPLIER --weights convnetquakeingv-lendb/ --use-max-stream --cudnn-lstm --save-best --save-all --log-to-file --verbose 2
python train.py convnetquakeingv speech8khz1k5_4 classification --epochs 250 --lr 0.001 --lr-reducer --early-stopping --name convnetquakeingv-stead-speech8khz1k5_4-$LR_MULTIPLIER --lr-multiplier $LR_MULTIPLIER --weights convnetquakeingv-stead/ --use-max-stream --cudnn-lstm --save-best --save-all --log-to-file --verbose 2
python train.py convnetquakeingv speech8khz9k_4 classification --epochs 250 --lr 0.001 --lr-reducer --early-stopping --name convnetquakeingv-stead-speech8khz9k_4-$LR_MULTIPLIER --lr-multiplier $LR_MULTIPLIER --weights convnetquakeingv-stead/ --use-max-stream --cudnn-lstm --save-best --save-all --log-to-file --verbose 2
python train.py convnetquakeingv speech8khz1k5_4 classification --epochs 250 --lr 0.001 --lr-reducer --early-stopping --name convnetquakeingv-emg-speech8khz1k5_4-$LR_MULTIPLIER --lr-multiplier $LR_MULTIPLIER --weights convnetquakeingv-emg/ --use-max-stream --cudnn-lstm --save-best --save-all --log-to-file --verbose 2
python train.py convnetquakeingv speech8khz9k_4 classification --epochs 250 --lr 0.001 --lr-reducer --early-stopping --name convnetquakeingv-emg-speech8khz9k_4-$LR_MULTIPLIER --lr-multiplier $LR_MULTIPLIER --weights convnetquakeingv-emg/ --use-max-stream --cudnn-lstm --save-best --save-all --log-to-file --verbose 2
python train.py convnetquakeingv speech8khz1k5_4 classification --epochs 250 --lr 0.001 --lr-reducer --early-stopping --name convnetquakeingv-sp500-speech8khz1k5_4-$LR_MULTIPLIER --lr-multiplier $LR_MULTIPLIER --weights convnetquakeingv-sp500/ --use-max-stream --cudnn-lstm --save-best --save-all --log-to-file --verbose 2
python train.py convnetquakeingv speech8khz9k_4 classification --epochs 250 --lr 0.001 --lr-reducer --early-stopping --name convnetquakeingv-sp500-speech8khz9k_4-$LR_MULTIPLIER --lr-multiplier $LR_MULTIPLIER --weights convnetquakeingv-sp500/ --use-max-stream --cudnn-lstm --save-best --save-all --log-to-file --verbose 2