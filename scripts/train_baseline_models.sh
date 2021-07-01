#!/bin/bash

python train.py magnet lomax regression --epochs 250 --lr 0.001 --lr-reducer --early-stopping --name magnet-lomax --use-max-stream --save-best --cudnn-lstm
python train.py magnet lendb regression --epochs 250 --lr 0.001 --lr-reducer --early-stopping --name magnet-lendb --use-max-stream --save-best --cudnn-lstm
python train.py magnet stead regression --epochs 250 --lr 0.001 --lr-reducer --early-stopping --name magnet-stead --save-best --cudnn-lstm
python train.py magnet speech8khz classification --epochs 250 --lr 0.001 --lr-reducer --early-stopping --name magnet-speech8khz --use-max-stream --cudnn-lstm --save-best
python train.py magnet emg classification --epochs 250 --lr 0.001 --lr-reducer --early-stopping --name magnet-emg --use-max-stream --save-best --cudnn-lstm
python train.py magnet sp500 regression --epochs 250 --lr 0.001 --lr-reducer --early-stopping --name magnet-sp500 --use-max-stream --save-best --cudnn-lstm
python train.py convnetquakeingv lomax regression --epochs 250 --lr 0.001 --lr-reducer --early-stopping --name convnetquakeingv-lomax --save-best --use-max-stream
python train.py convnetquakeingv lendb regression --epochs 250 --lr 0.001 --lr-reducer --early-stopping --name convnetquakeingv-lendb --save-best --use-max-stream
python train.py convnetquakeingv stead regression --epochs 250 --lr 0.001 --lr-reducer --early-stopping --name convnetquakeingv-stead --save-best
python train.py convnetquakeingv_speech8khz speech8khz classification --epochs 250 --lr 0.001 --lr-reducer --early-stopping --name convnetquakeingv-speech8khz --save-best --use-max-stream
python train.py convnetquakeingv_emg emg classification --epochs 250 --lr 0.001 --lr-reducer --early-stopping --name convnetquakeingv-emg --save-best --use-max-stream
python train.py convnetquakeingv_sp500 sp500 regression --epochs 250 --lr 0.001 --lr-reducer --early-stopping --name convnetquakeingv-sp500 --save-best --use-max-stream
python train.py mlstm_fcn lomax regression --epochs 250 --lr 0.001 --lr-reducer --early-stopping --name mlstm_fcn-lomax --use-max-stream --save-best
python train.py mlstm_fcn lendb regression --epochs 250 --lr 0.001 --lr-reducer --early-stopping --name mlstm_fcn-lendb --use-max-stream --save-best
python train.py mlstm_fcn stead regression --epochs 250 --lr 0.001 --lr-reducer --early-stopping --name mlstm_fcn-stead --save-best
python train.py mlstm_fcn speech8khz classification --epochs 250 --lr 0.001 --lr-reducer --early-stopping --name mlstm_fcn-speech8khz --save-best --use-max-stream
python train.py mlstm_fcn emg classification --epochs 250 --lr 0.001 --lr-reducer --early-stopping --name mlstm_fcn-emg --use-max-stream --save-best
python train.py mlstm_fcn sp500 regression --epochs 250 --lr 0.001 --lr-reducer --early-stopping --name mlstm_fcn-sp500 --use-max-stream --save-best
python train.py tcn lomax regression --epochs 250 --lr 0.001 --lr-reducer --early-stopping --name tcn-lomax --use-max-stream --save-best
python train.py tcn lendb regression --epochs 250 --lr 0.001 --lr-reducer --early-stopping --name tcn-lendb --use-max-stream --save-best
python train.py tcn stead regression --epochs 250 --lr 0.001 --lr-reducer --early-stopping --name tcn-stead --save-best
python train.py tcn speech8khz classification --epochs 250 --lr 0.001 --lr-reducer --early-stopping --name tcn-speech8khz --use-max-stream --save-best
python train.py tcn emg classification --epochs 250 --lr 0.001 --lr-reducer --early-stopping --name tcn-emg --use-max-stream --save-best
python train.py tcn sp500 regression --epochs 250 --lr 0.001 --lr-reducer --early-stopping --name tcn-sp500 --use-max-stream --save-best