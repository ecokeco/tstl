#!/bin/bash

python train.py mlstm_fcn emg1k5_6 classification --epochs 250 --lr 0.001 --lr-reducer --early-stopping --name mlstm_fcn-lomax-emg1k5_6-$LR_MULTIPLIER --lr-multiplier $LR_MULTIPLIER --weights mlstm_fcn-lomax/ --use-max-stream --cudnn-lstm --save-best --save-all --log-to-file --verbose 2
python train.py mlstm_fcn emg9k_6 classification --epochs 250 --lr 0.001 --lr-reducer --early-stopping --name mlstm_fcn-lomax-emg9k_6-$LR_MULTIPLIER --lr-multiplier $LR_MULTIPLIER --weights mlstm_fcn-lomax/ --use-max-stream --cudnn-lstm --save-best --save-all --log-to-file --verbose 2
python train.py mlstm_fcn emg1k5_6 classification --epochs 250 --lr 0.001 --lr-reducer --early-stopping --name mlstm_fcn-lendb-emg1k5_6-$LR_MULTIPLIER --lr-multiplier $LR_MULTIPLIER --weights mlstm_fcn-lendb/ --use-max-stream --cudnn-lstm --save-best --save-all --log-to-file --verbose 2
python train.py mlstm_fcn emg9k_6 classification --epochs 250 --lr 0.001 --lr-reducer --early-stopping --name mlstm_fcn-lendb-emg9k_6-$LR_MULTIPLIER --lr-multiplier $LR_MULTIPLIER --weights mlstm_fcn-lendb/ --use-max-stream --cudnn-lstm --save-best --save-all --log-to-file --verbose 2
python train.py mlstm_fcn emg1k5_6 classification --epochs 250 --lr 0.001 --lr-reducer --early-stopping --name mlstm_fcn-stead-emg1k5_6-$LR_MULTIPLIER --lr-multiplier $LR_MULTIPLIER --weights mlstm_fcn-stead/ --use-max-stream --cudnn-lstm --save-best --save-all --log-to-file --verbose 2
python train.py mlstm_fcn emg9k_6 classification --epochs 250 --lr 0.001 --lr-reducer --early-stopping --name mlstm_fcn-stead-emg9k_6-$LR_MULTIPLIER --lr-multiplier $LR_MULTIPLIER --weights mlstm_fcn-stead/ --use-max-stream --cudnn-lstm --save-best --save-all --log-to-file --verbose 2
python train.py mlstm_fcn emg1k5_6 classification --epochs 250 --lr 0.001 --lr-reducer --early-stopping --name mlstm_fcn-speech8khz-emg1k5_6-$LR_MULTIPLIER --lr-multiplier $LR_MULTIPLIER --weights mlstm_fcn-speech8khz/ --use-max-stream --cudnn-lstm --save-best --save-all --log-to-file --verbose 2
python train.py mlstm_fcn emg9k_6 classification --epochs 250 --lr 0.001 --lr-reducer --early-stopping --name mlstm_fcn-speech8khz-emg9k_6-$LR_MULTIPLIER --lr-multiplier $LR_MULTIPLIER --weights mlstm_fcn-speech8khz/ --use-max-stream --cudnn-lstm --save-best --save-all --log-to-file --verbose 2
python train.py mlstm_fcn emg1k5_6 classification --epochs 250 --lr 0.001 --lr-reducer --early-stopping --name mlstm_fcn-sp500-emg1k5_6-$LR_MULTIPLIER --lr-multiplier $LR_MULTIPLIER --weights mlstm_fcn-sp500/ --use-max-stream --cudnn-lstm --save-best --save-all --log-to-file --verbose 2
python train.py mlstm_fcn emg9k_6 classification --epochs 250 --lr 0.001 --lr-reducer --early-stopping --name mlstm_fcn-sp500-emg9k_6-$LR_MULTIPLIER --lr-multiplier $LR_MULTIPLIER --weights mlstm_fcn-sp500/ --use-max-stream --cudnn-lstm --save-best --save-all --log-to-file --verbose 2