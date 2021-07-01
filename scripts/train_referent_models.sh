#!/bin/bash

# 1st run
echo "Training ConvNetQuakeINGV referent model"
python train.py convnetquakeingv lomax1k5_0 regression --epochs 250 --lr 0.001 --lr-reducer --early-stopping --name convnetquakeingv-lomax1k5_0 --use-max-stream --cudnn-lstm --save-best --save-all --log-to-file --verbose 2
python train.py convnetquakeingv lomax9k_0 regression --epochs 250 --lr 0.001 --lr-reducer --early-stopping --name convnetquakeingv-lomax9k_0 --use-max-stream --cudnn-lstm --save-best --save-all --log-to-file --verbose 2
python train.py convnetquakeingv lendb1k5_0 regression --epochs 250 --lr 0.001 --lr-reducer --early-stopping --name convnetquakeingv-lendb1k5_0 --use-max-stream --cudnn-lstm --save-best --save-all --log-to-file --verbose 2
python train.py convnetquakeingv lendb9k_0 regression --epochs 250 --lr 0.001 --lr-reducer --early-stopping --name convnetquakeingv-lendb9k_0 --use-max-stream --cudnn-lstm --save-best --save-all --log-to-file --verbose 2
python train.py convnetquakeingv stead1k5_0 regression --epochs 250 --lr 0.001 --lr-reducer --early-stopping --name convnetquakeingv-stead1k5_0 --cudnn-lstm --save-best --save-all --log-to-file --verbose 2
python train.py convnetquakeingv stead9k_0 regression --epochs 250 --lr 0.001 --lr-reducer --early-stopping --name convnetquakeingv-stead9k_0 --cudnn-lstm --save-best --save-all --log-to-file --verbose 2

python train.py convnetquakeingv speech8khz1k5_0 regression --epochs 250 --lr 0.001 --lr-reducer --early-stopping --name convnetquakeingv-speech8khz1k5_0 --use-max-stream --cudnn-lstm --save-best --save-all --log-to-file --verbose 2
python train.py convnetquakeingv speech8khz9k_0 regression --epochs 250 --lr 0.001 --lr-reducer --early-stopping --name convnetquakeingv-speech8khz9k_0 --use-max-stream --cudnn-lstm --save-best --save-all --log-to-file --verbose 2
python train.py convnetquakeingv emg1k5_0 regression --epochs 250 --lr 0.001 --lr-reducer --early-stopping --name convnetquakeingv-emg1k5_0 --use-max-stream --cudnn-lstm --save-best --save-all --log-to-file --verbose 2
python train.py convnetquakeingv emg9k_0 regression --epochs 250 --lr 0.001 --lr-reducer --early-stopping --name convnetquakeingv-emg9k_0 --use-max-stream --cudnn-lstm --save-best --save-all --log-to-file --verbose 2
python train.py convnetquakeingv sp5001k5_0 regression --epochs 250 --lr 0.001 --lr-reducer --early-stopping --name convnetquakeingv-sp5001k5_0 --use-max-stream --cudnn-lstm --save-best --save-all --log-to-file --verbose 2
python train.py convnetquakeingv sp5009k_0 regression --epochs 250 --lr 0.001 --lr-reducer --early-stopping --name convnetquakeingv-sp5009k_0 --use-max-stream --cudnn-lstm --save-best --save-all --log-to-file --verbose 2

echo "Training MagNet referent model"
python train.py magnet lomax1k5_0 regression --epochs 250 --lr 0.001 --lr-reducer --early-stopping --name magnet-lomax1k5_0 --use-max-stream --cudnn-lstm --save-best --save-all --log-to-file --verbose 2
python train.py magnet lomax9k_0 regression --epochs 250 --lr 0.001 --lr-reducer --early-stopping --name magnet-lomax9k_0 --use-max-stream --cudnn-lstm --save-best --save-all --log-to-file --verbose 2
python train.py magnet lendb1k5_0 regression --epochs 250 --lr 0.001 --lr-reducer --early-stopping --name magnet-lendb1k5_0 --use-max-stream --cudnn-lstm --save-best --save-all --log-to-file --verbose 2
python train.py magnet lendb9k_0 regression --epochs 250 --lr 0.001 --lr-reducer --early-stopping --name magnet-lendb9k_0 --use-max-stream --cudnn-lstm --save-best --save-all --log-to-file --verbose 2
python train.py magnet stead1k5_0 regression --epochs 250 --lr 0.001 --lr-reducer --early-stopping --name magnet-stead1k5_0 --cudnn-lstm --save-best --save-all --log-to-file --verbose 2
python train.py magnet stead9k_0 regression --epochs 250 --lr 0.001 --lr-reducer --early-stopping --name magnet-stead9k_0 --cudnn-lstm --save-best --save-all --log-to-file --verbose 2

python train.py magnet speech8khz1k5_0 regression --epochs 250 --lr 0.001 --lr-reducer --early-stopping --name magnet-speech8khz1k5_0 --use-max-stream --cudnn-lstm --save-best --save-all --log-to-file --verbose 2
python train.py magnet speech8khz9k_0 regression --epochs 250 --lr 0.001 --lr-reducer --early-stopping --name magnet-speech8khz9k_0 --use-max-stream --cudnn-lstm --save-best --save-all --log-to-file --verbose 2
python train.py magnet emg1k5_0 regression --epochs 250 --lr 0.001 --lr-reducer --early-stopping --name magnet-emg1k5_0 --use-max-stream --cudnn-lstm --save-best --save-all --log-to-file --verbose 2
python train.py magnet emg9k_0 regression --epochs 250 --lr 0.001 --lr-reducer --early-stopping --name magnet-emg9k_0 --use-max-stream --cudnn-lstm --save-best --save-all --log-to-file --verbose 2
python train.py magnet sp5001k5_0 regression --epochs 250 --lr 0.001 --lr-reducer --early-stopping --name magnet-sp5001k5_0 --use-max-stream --cudnn-lstm --save-best --save-all --log-to-file --verbose 2
python train.py magnet sp5009k_0 regression --epochs 250 --lr 0.001 --lr-reducer --early-stopping --name magnet-sp5009k_0 --use-max-stream --cudnn-lstm --save-best --save-all --log-to-file --verbose 2

echo "Training MLSTM_FCN referent model"
python train.py mlstm_fcn lomax1k5_0 regression --epochs 250 --lr 0.001 --lr-reducer --early-stopping --name mlstm_fcn-lomax1k5_0 --use-max-stream --cudnn-lstm --save-best --save-all --log-to-file --verbose 2
python train.py mlstm_fcn lomax9k_0 regression --epochs 250 --lr 0.001 --lr-reducer --early-stopping --name mlstm_fcn-lomax9k_0 --use-max-stream --cudnn-lstm --save-best --save-all --log-to-file --verbose 2
python train.py mlstm_fcn lendb1k5_0 regression --epochs 250 --lr 0.001 --lr-reducer --early-stopping --name mlstm_fcn-lendb1k5_0 --use-max-stream --cudnn-lstm --save-best --save-all --log-to-file --verbose 2
python train.py mlstm_fcn lendb9k_0 regression --epochs 250 --lr 0.001 --lr-reducer --early-stopping --name mlstm_fcn-lendb9k_0 --use-max-stream --cudnn-lstm --save-best --save-all --log-to-file --verbose 2
python train.py mlstm_fcn stead1k5_0 regression --epochs 250 --lr 0.001 --lr-reducer --early-stopping --name mlstm_fcn-stead1k5_0 --cudnn-lstm --save-best --save-all --log-to-file --verbose 2
python train.py mlstm_fcn stead9k_0 regression --epochs 250 --lr 0.001 --lr-reducer --early-stopping --name mlstm_fcn-stead9k_0 --cudnn-lstm --save-best --save-all --log-to-file --verbose 2

python train.py mlstm_fcn speech8khz1k5_0 regression --epochs 250 --lr 0.001 --lr-reducer --early-stopping --name mlstm_fcn-speech8khz1k5_0 --use-max-stream --cudnn-lstm --save-best --save-all --log-to-file --verbose 2
python train.py mlstm_fcn speech8khz9k_0 regression --epochs 250 --lr 0.001 --lr-reducer --early-stopping --name mlstm_fcn-speech8khz9k_0 --use-max-stream --cudnn-lstm --save-best --save-all --log-to-file --verbose 2
python train.py mlstm_fcn emg1k5_0 regression --epochs 250 --lr 0.001 --lr-reducer --early-stopping --name mlstm_fcn-emg1k5_0 --use-max-stream --cudnn-lstm --save-best --save-all --log-to-file --verbose 2
python train.py mlstm_fcn emg9k_0 regression --epochs 250 --lr 0.001 --lr-reducer --early-stopping --name mlstm_fcn-emg9k_0 --use-max-stream --cudnn-lstm --save-best --save-all --log-to-file --verbose 2
python train.py mlstm_fcn sp5001k5_0 regression --epochs 250 --lr 0.001 --lr-reducer --early-stopping --name mlstm_fcn-sp5001k5_0 --use-max-stream --cudnn-lstm --save-best --save-all --log-to-file --verbose 2
python train.py mlstm_fcn sp5009k_0 regression --epochs 250 --lr 0.001 --lr-reducer --early-stopping --name mlstm_fcn-sp5009k_0 --use-max-stream --cudnn-lstm --save-best --save-all --log-to-file --verbose 2

echo "Training TCN referent model"
python train.py tcn lomax1k5_0 regression --epochs 250 --lr 0.001 --lr-reducer --early-stopping --name tcn-lomax1k5_0 --use-max-stream --cudnn-lstm --save-best --save-all --log-to-file --verbose 2
python train.py tcn lomax9k_0 regression --epochs 250 --lr 0.001 --lr-reducer --early-stopping --name tcn-lomax9k_0 --use-max-stream --cudnn-lstm --save-best --save-all --log-to-file --verbose 2
python train.py tcn lendb1k5_0 regression --epochs 250 --lr 0.001 --lr-reducer --early-stopping --name tcn-lendb1k5_0 --use-max-stream --cudnn-lstm --save-best --save-all --log-to-file --verbose 2
python train.py tcn lendb9k_0 regression --epochs 250 --lr 0.001 --lr-reducer --early-stopping --name tcn-lendb9k_0 --use-max-stream --cudnn-lstm --save-best --save-all --log-to-file --verbose 2
python train.py tcn stead1k5_0 regression --epochs 250 --lr 0.001 --lr-reducer --early-stopping --name tcn-stead1k5_0 --cudnn-lstm --save-best --save-all --log-to-file --verbose 2
python train.py tcn stead9k_0 regression --epochs 250 --lr 0.001 --lr-reducer --early-stopping --name tcn-stead9k_0 --cudnn-lstm --save-best --save-all --log-to-file --verbose 2

python train.py tcn speech8khz1k5_0 regression --epochs 250 --lr 0.001 --lr-reducer --early-stopping --name tcn-speech8khz1k5_0 --use-max-stream --cudnn-lstm --save-best --save-all --log-to-file --verbose 2
python train.py tcn speech8khz9k_0 regression --epochs 250 --lr 0.001 --lr-reducer --early-stopping --name tcn-speech8khz9k_0 --use-max-stream --cudnn-lstm --save-best --save-all --log-to-file --verbose 2
python train.py tcn emg1k5_0 regression --epochs 250 --lr 0.001 --lr-reducer --early-stopping --name tcn-emg1k5_0 --use-max-stream --cudnn-lstm --save-best --save-all --log-to-file --verbose 2
python train.py tcn emg9k_0 regression --epochs 250 --lr 0.001 --lr-reducer --early-stopping --name tcn-emg9k_0 --use-max-stream --cudnn-lstm --save-best --save-all --log-to-file --verbose 2
python train.py tcn sp5001k5_0 regression --epochs 250 --lr 0.001 --lr-reducer --early-stopping --name tcn-sp5001k5_0 --use-max-stream --cudnn-lstm --save-best --save-all --log-to-file --verbose 2
python train.py tcn sp5009k_0 regression --epochs 250 --lr 0.001 --lr-reducer --early-stopping --name tcn-sp5009k_0 --use-max-stream --cudnn-lstm --save-best --save-all --log-to-file --verbose 2

# 2nd run
echo "Training ConvNetQuakeINGV referent model"
python train.py convnetquakeingv lomax1k5_1 regression --epochs 250 --lr 0.001 --lr-reducer --early-stopping --name convnetquakeingv-lomax1k5_1 --use-max-stream --cudnn-lstm --save-best --save-all --log-to-file --verbose 2
python train.py convnetquakeingv lomax9k_1 regression --epochs 250 --lr 0.001 --lr-reducer --early-stopping --name convnetquakeingv-lomax9k_1 --use-max-stream --cudnn-lstm --save-best --save-all --log-to-file --verbose 2
python train.py convnetquakeingv lendb1k5_1 regression --epochs 250 --lr 0.001 --lr-reducer --early-stopping --name convnetquakeingv-lendb1k5_1 --use-max-stream --cudnn-lstm --save-best --save-all --log-to-file --verbose 2
python train.py convnetquakeingv lendb9k_1 regression --epochs 250 --lr 0.001 --lr-reducer --early-stopping --name convnetquakeingv-lendb9k_1 --use-max-stream --cudnn-lstm --save-best --save-all --log-to-file --verbose 2
python train.py convnetquakeingv stead1k5_1 regression --epochs 250 --lr 0.001 --lr-reducer --early-stopping --name convnetquakeingv-stead1k5_1 --cudnn-lstm --save-best --save-all --log-to-file --verbose 2
python train.py convnetquakeingv stead9k_1 regression --epochs 250 --lr 0.001 --lr-reducer --early-stopping --name convnetquakeingv-stead9k_1 --cudnn-lstm --save-best --save-all --log-to-file --verbose 2

python train.py convnetquakeingv speech8khz1k5_1 regression --epochs 250 --lr 0.001 --lr-reducer --early-stopping --name convnetquakeingv-speech8khz1k5_1 --use-max-stream --cudnn-lstm --save-best --save-all --log-to-file --verbose 2
python train.py convnetquakeingv speech8khz9k_1 regression --epochs 250 --lr 0.001 --lr-reducer --early-stopping --name convnetquakeingv-speech8khz9k_1 --use-max-stream --cudnn-lstm --save-best --save-all --log-to-file --verbose 2
python train.py convnetquakeingv emg1k5_1 regression --epochs 250 --lr 0.001 --lr-reducer --early-stopping --name convnetquakeingv-emg1k5_1 --use-max-stream --cudnn-lstm --save-best --save-all --log-to-file --verbose 2
python train.py convnetquakeingv emg9k_1 regression --epochs 250 --lr 0.001 --lr-reducer --early-stopping --name convnetquakeingv-emg9k_1 --use-max-stream --cudnn-lstm --save-best --save-all --log-to-file --verbose 2
python train.py convnetquakeingv sp5001k5_1 regression --epochs 250 --lr 0.001 --lr-reducer --early-stopping --name convnetquakeingv-sp5001k5_1 --use-max-stream --cudnn-lstm --save-best --save-all --log-to-file --verbose 2
python train.py convnetquakeingv sp5009k_1 regression --epochs 250 --lr 0.001 --lr-reducer --early-stopping --name convnetquakeingv-sp5009k_1 --use-max-stream --cudnn-lstm --save-best --save-all --log-to-file --verbose 2

echo "Training MagNet referent model"
python train.py magnet lomax1k5_1 regression --epochs 250 --lr 0.001 --lr-reducer --early-stopping --name magnet-lomax1k5_1 --use-max-stream --cudnn-lstm --save-best --save-all --log-to-file --verbose 2
python train.py magnet lomax9k_1 regression --epochs 250 --lr 0.001 --lr-reducer --early-stopping --name magnet-lomax9k_1 --use-max-stream --cudnn-lstm --save-best --save-all --log-to-file --verbose 2
python train.py magnet lendb1k5_1 regression --epochs 250 --lr 0.001 --lr-reducer --early-stopping --name magnet-lendb1k5_1 --use-max-stream --cudnn-lstm --save-best --save-all --log-to-file --verbose 2
python train.py magnet lendb9k_1 regression --epochs 250 --lr 0.001 --lr-reducer --early-stopping --name magnet-lendb9k_1 --use-max-stream --cudnn-lstm --save-best --save-all --log-to-file --verbose 2
python train.py magnet stead1k5_1 regression --epochs 250 --lr 0.001 --lr-reducer --early-stopping --name magnet-stead1k5_1 --cudnn-lstm --save-best --save-all --log-to-file --verbose 2
python train.py magnet stead9k_1 regression --epochs 250 --lr 0.001 --lr-reducer --early-stopping --name magnet-stead9k_1 --cudnn-lstm --save-best --save-all --log-to-file --verbose 2

python train.py magnet speech8khz1k5_1 regression --epochs 250 --lr 0.001 --lr-reducer --early-stopping --name magnet-speech8khz1k5_1 --use-max-stream --cudnn-lstm --save-best --save-all --log-to-file --verbose 2
python train.py magnet speech8khz9k_1 regression --epochs 250 --lr 0.001 --lr-reducer --early-stopping --name magnet-speech8khz9k_1 --use-max-stream --cudnn-lstm --save-best --save-all --log-to-file --verbose 2
python train.py magnet emg1k5_1 regression --epochs 250 --lr 0.001 --lr-reducer --early-stopping --name magnet-emg1k5_1 --use-max-stream --cudnn-lstm --save-best --save-all --log-to-file --verbose 2
python train.py magnet emg9k_1 regression --epochs 250 --lr 0.001 --lr-reducer --early-stopping --name magnet-emg9k_1 --use-max-stream --cudnn-lstm --save-best --save-all --log-to-file --verbose 2
python train.py magnet sp5001k5_1 regression --epochs 250 --lr 0.001 --lr-reducer --early-stopping --name magnet-sp5001k5_1 --use-max-stream --cudnn-lstm --save-best --save-all --log-to-file --verbose 2
python train.py magnet sp5009k_1 regression --epochs 250 --lr 0.001 --lr-reducer --early-stopping --name magnet-sp5009k_1 --use-max-stream --cudnn-lstm --save-best --save-all --log-to-file --verbose 2

echo "Training MLSTM_FCN referent model"
python train.py mlstm_fcn lomax1k5_1 regression --epochs 250 --lr 0.001 --lr-reducer --early-stopping --name mlstm_fcn-lomax1k5_1 --use-max-stream --cudnn-lstm --save-best --save-all --log-to-file --verbose 2
python train.py mlstm_fcn lomax9k_1 regression --epochs 250 --lr 0.001 --lr-reducer --early-stopping --name mlstm_fcn-lomax9k_1 --use-max-stream --cudnn-lstm --save-best --save-all --log-to-file --verbose 2
python train.py mlstm_fcn lendb1k5_1 regression --epochs 250 --lr 0.001 --lr-reducer --early-stopping --name mlstm_fcn-lendb1k5_1 --use-max-stream --cudnn-lstm --save-best --save-all --log-to-file --verbose 2
python train.py mlstm_fcn lendb9k_1 regression --epochs 250 --lr 0.001 --lr-reducer --early-stopping --name mlstm_fcn-lendb9k_1 --use-max-stream --cudnn-lstm --save-best --save-all --log-to-file --verbose 2
python train.py mlstm_fcn stead1k5_1 regression --epochs 250 --lr 0.001 --lr-reducer --early-stopping --name mlstm_fcn-stead1k5_1 --cudnn-lstm --save-best --save-all --log-to-file --verbose 2
python train.py mlstm_fcn stead9k_1 regression --epochs 250 --lr 0.001 --lr-reducer --early-stopping --name mlstm_fcn-stead9k_1 --cudnn-lstm --save-best --save-all --log-to-file --verbose 2

python train.py mlstm_fcn speech8khz1k5_1 regression --epochs 250 --lr 0.001 --lr-reducer --early-stopping --name mlstm_fcn-speech8khz1k5_1 --use-max-stream --cudnn-lstm --save-best --save-all --log-to-file --verbose 2
python train.py mlstm_fcn speech8khz9k_1 regression --epochs 250 --lr 0.001 --lr-reducer --early-stopping --name mlstm_fcn-speech8khz9k_1 --use-max-stream --cudnn-lstm --save-best --save-all --log-to-file --verbose 2
python train.py mlstm_fcn emg1k5_1 regression --epochs 250 --lr 0.001 --lr-reducer --early-stopping --name mlstm_fcn-emg1k5_1 --use-max-stream --cudnn-lstm --save-best --save-all --log-to-file --verbose 2
python train.py mlstm_fcn emg9k_1 regression --epochs 250 --lr 0.001 --lr-reducer --early-stopping --name mlstm_fcn-emg9k_1 --use-max-stream --cudnn-lstm --save-best --save-all --log-to-file --verbose 2
python train.py mlstm_fcn sp5001k5_1 regression --epochs 250 --lr 0.001 --lr-reducer --early-stopping --name mlstm_fcn-sp5001k5_1 --use-max-stream --cudnn-lstm --save-best --save-all --log-to-file --verbose 2
python train.py mlstm_fcn sp5009k_1 regression --epochs 250 --lr 0.001 --lr-reducer --early-stopping --name mlstm_fcn-sp5009k_1 --use-max-stream --cudnn-lstm --save-best --save-all --log-to-file --verbose 2

echo "Training TCN referent model"
python train.py tcn lomax1k5_1 regression --epochs 250 --lr 0.001 --lr-reducer --early-stopping --name tcn-lomax1k5_1 --use-max-stream --cudnn-lstm --save-best --save-all --log-to-file --verbose 2
python train.py tcn lomax9k_1 regression --epochs 250 --lr 0.001 --lr-reducer --early-stopping --name tcn-lomax9k_1 --use-max-stream --cudnn-lstm --save-best --save-all --log-to-file --verbose 2
python train.py tcn lendb1k5_1 regression --epochs 250 --lr 0.001 --lr-reducer --early-stopping --name tcn-lendb1k5_1 --use-max-stream --cudnn-lstm --save-best --save-all --log-to-file --verbose 2
python train.py tcn lendb9k_1 regression --epochs 250 --lr 0.001 --lr-reducer --early-stopping --name tcn-lendb9k_1 --use-max-stream --cudnn-lstm --save-best --save-all --log-to-file --verbose 2
python train.py tcn stead1k5_1 regression --epochs 250 --lr 0.001 --lr-reducer --early-stopping --name tcn-stead1k5_1 --cudnn-lstm --save-best --save-all --log-to-file --verbose 2
python train.py tcn stead9k_1 regression --epochs 250 --lr 0.001 --lr-reducer --early-stopping --name tcn-stead9k_1 --cudnn-lstm --save-best --save-all --log-to-file --verbose 2

python train.py tcn speech8khz1k5_1 regression --epochs 250 --lr 0.001 --lr-reducer --early-stopping --name tcn-speech8khz1k5_1 --use-max-stream --cudnn-lstm --save-best --save-all --log-to-file --verbose 2
python train.py tcn speech8khz9k_1 regression --epochs 250 --lr 0.001 --lr-reducer --early-stopping --name tcn-speech8khz9k_1 --use-max-stream --cudnn-lstm --save-best --save-all --log-to-file --verbose 2
python train.py tcn emg1k5_1 regression --epochs 250 --lr 0.001 --lr-reducer --early-stopping --name tcn-emg1k5_1 --use-max-stream --cudnn-lstm --save-best --save-all --log-to-file --verbose 2
python train.py tcn emg9k_1 regression --epochs 250 --lr 0.001 --lr-reducer --early-stopping --name tcn-emg9k_1 --use-max-stream --cudnn-lstm --save-best --save-all --log-to-file --verbose 2
python train.py tcn sp5001k5_1 regression --epochs 250 --lr 0.001 --lr-reducer --early-stopping --name tcn-sp5001k5_1 --use-max-stream --cudnn-lstm --save-best --save-all --log-to-file --verbose 2
python train.py tcn sp5009k_1 regression --epochs 250 --lr 0.001 --lr-reducer --early-stopping --name tcn-sp5009k_1 --use-max-stream --cudnn-lstm --save-best --save-all --log-to-file --verbose 2

# 3rd run
echo "Training ConvNetQuakeINGV referent model"
python train.py convnetquakeingv lomax1k5_2 regression --epochs 250 --lr 0.001 --lr-reducer --early-stopping --name convnetquakeingv-lomax1k5_2 --use-max-stream --cudnn-lstm --save-best --save-all --log-to-file --verbose 2
python train.py convnetquakeingv lomax9k_2 regression --epochs 250 --lr 0.001 --lr-reducer --early-stopping --name convnetquakeingv-lomax9k_2 --use-max-stream --cudnn-lstm --save-best --save-all --log-to-file --verbose 2
python train.py convnetquakeingv lendb1k5_2 regression --epochs 250 --lr 0.001 --lr-reducer --early-stopping --name convnetquakeingv-lendb1k5_2 --use-max-stream --cudnn-lstm --save-best --save-all --log-to-file --verbose 2
python train.py convnetquakeingv lendb9k_2 regression --epochs 250 --lr 0.001 --lr-reducer --early-stopping --name convnetquakeingv-lendb9k_2 --use-max-stream --cudnn-lstm --save-best --save-all --log-to-file --verbose 2
python train.py convnetquakeingv stead1k5_2 regression --epochs 250 --lr 0.001 --lr-reducer --early-stopping --name convnetquakeingv-stead1k5_2 --cudnn-lstm --save-best --save-all --log-to-file --verbose 2
python train.py convnetquakeingv stead9k_2 regression --epochs 250 --lr 0.001 --lr-reducer --early-stopping --name convnetquakeingv-stead9k_2 --cudnn-lstm --save-best --save-all --log-to-file --verbose 2

python train.py convnetquakeingv speech8khz1k5_2 regression --epochs 250 --lr 0.001 --lr-reducer --early-stopping --name convnetquakeingv-speech8khz1k5_2 --use-max-stream --cudnn-lstm --save-best --save-all --log-to-file --verbose 2
python train.py convnetquakeingv speech8khz9k_2 regression --epochs 250 --lr 0.001 --lr-reducer --early-stopping --name convnetquakeingv-speech8khz9k_2 --use-max-stream --cudnn-lstm --save-best --save-all --log-to-file --verbose 2
python train.py convnetquakeingv emg1k5_2 regression --epochs 250 --lr 0.001 --lr-reducer --early-stopping --name convnetquakeingv-emg1k5_2 --use-max-stream --cudnn-lstm --save-best --save-all --log-to-file --verbose 2
python train.py convnetquakeingv emg9k_2 regression --epochs 250 --lr 0.001 --lr-reducer --early-stopping --name convnetquakeingv-emg9k_2 --use-max-stream --cudnn-lstm --save-best --save-all --log-to-file --verbose 2
python train.py convnetquakeingv sp5001k5_2 regression --epochs 250 --lr 0.001 --lr-reducer --early-stopping --name convnetquakeingv-sp5001k5_2 --use-max-stream --cudnn-lstm --save-best --save-all --log-to-file --verbose 2
python train.py convnetquakeingv sp5009k_2 regression --epochs 250 --lr 0.001 --lr-reducer --early-stopping --name convnetquakeingv-sp5009k_2 --use-max-stream --cudnn-lstm --save-best --save-all --log-to-file --verbose 2

echo "Training MagNet referent model"
python train.py magnet lomax1k5_2 regression --epochs 250 --lr 0.001 --lr-reducer --early-stopping --name magnet-lomax1k5_2 --use-max-stream --cudnn-lstm --save-best --save-all --log-to-file --verbose 2
python train.py magnet lomax9k_2 regression --epochs 250 --lr 0.001 --lr-reducer --early-stopping --name magnet-lomax9k_2 --use-max-stream --cudnn-lstm --save-best --save-all --log-to-file --verbose 2
python train.py magnet lendb1k5_2 regression --epochs 250 --lr 0.001 --lr-reducer --early-stopping --name magnet-lendb1k5_2 --use-max-stream --cudnn-lstm --save-best --save-all --log-to-file --verbose 2
python train.py magnet lendb9k_2 regression --epochs 250 --lr 0.001 --lr-reducer --early-stopping --name magnet-lendb9k_2 --use-max-stream --cudnn-lstm --save-best --save-all --log-to-file --verbose 2
python train.py magnet stead1k5_2 regression --epochs 250 --lr 0.001 --lr-reducer --early-stopping --name magnet-stead1k5_2 --cudnn-lstm --save-best --save-all --log-to-file --verbose 2
python train.py magnet stead9k_2 regression --epochs 250 --lr 0.001 --lr-reducer --early-stopping --name magnet-stead9k_2 --cudnn-lstm --save-best --save-all --log-to-file --verbose 2

python train.py magnet speech8khz1k5_2 regression --epochs 250 --lr 0.001 --lr-reducer --early-stopping --name magnet-speech8khz1k5_2 --use-max-stream --cudnn-lstm --save-best --save-all --log-to-file --verbose 2
python train.py magnet speech8khz9k_2 regression --epochs 250 --lr 0.001 --lr-reducer --early-stopping --name magnet-speech8khz9k_2 --use-max-stream --cudnn-lstm --save-best --save-all --log-to-file --verbose 2
python train.py magnet emg1k5_2 regression --epochs 250 --lr 0.001 --lr-reducer --early-stopping --name magnet-emg1k5_2 --use-max-stream --cudnn-lstm --save-best --save-all --log-to-file --verbose 2
python train.py magnet emg9k_2 regression --epochs 250 --lr 0.001 --lr-reducer --early-stopping --name magnet-emg9k_2 --use-max-stream --cudnn-lstm --save-best --save-all --log-to-file --verbose 2
python train.py magnet sp5001k5_2 regression --epochs 250 --lr 0.001 --lr-reducer --early-stopping --name magnet-sp5001k5_2 --use-max-stream --cudnn-lstm --save-best --save-all --log-to-file --verbose 2
python train.py magnet sp5009k_2 regression --epochs 250 --lr 0.001 --lr-reducer --early-stopping --name magnet-sp5009k_2 --use-max-stream --cudnn-lstm --save-best --save-all --log-to-file --verbose 2

echo "Training MLSTM_FCN referent model"
python train.py mlstm_fcn lomax1k5_2 regression --epochs 250 --lr 0.001 --lr-reducer --early-stopping --name mlstm_fcn-lomax1k5_2 --use-max-stream --cudnn-lstm --save-best --save-all --log-to-file --verbose 2
python train.py mlstm_fcn lomax9k_2 regression --epochs 250 --lr 0.001 --lr-reducer --early-stopping --name mlstm_fcn-lomax9k_2 --use-max-stream --cudnn-lstm --save-best --save-all --log-to-file --verbose 2
python train.py mlstm_fcn lendb1k5_2 regression --epochs 250 --lr 0.001 --lr-reducer --early-stopping --name mlstm_fcn-lendb1k5_2 --use-max-stream --cudnn-lstm --save-best --save-all --log-to-file --verbose 2
python train.py mlstm_fcn lendb9k_2 regression --epochs 250 --lr 0.001 --lr-reducer --early-stopping --name mlstm_fcn-lendb9k_2 --use-max-stream --cudnn-lstm --save-best --save-all --log-to-file --verbose 2
python train.py mlstm_fcn stead1k5_2 regression --epochs 250 --lr 0.001 --lr-reducer --early-stopping --name mlstm_fcn-stead1k5_2 --cudnn-lstm --save-best --save-all --log-to-file --verbose 2
python train.py mlstm_fcn stead9k_2 regression --epochs 250 --lr 0.001 --lr-reducer --early-stopping --name mlstm_fcn-stead9k_2 --cudnn-lstm --save-best --save-all --log-to-file --verbose 2

python train.py mlstm_fcn speech8khz1k5_2 regression --epochs 250 --lr 0.001 --lr-reducer --early-stopping --name mlstm_fcn-speech8khz1k5_2 --use-max-stream --cudnn-lstm --save-best --save-all --log-to-file --verbose 2
python train.py mlstm_fcn speech8khz9k_2 regression --epochs 250 --lr 0.001 --lr-reducer --early-stopping --name mlstm_fcn-speech8khz9k_2 --use-max-stream --cudnn-lstm --save-best --save-all --log-to-file --verbose 2
python train.py mlstm_fcn emg1k5_2 regression --epochs 250 --lr 0.001 --lr-reducer --early-stopping --name mlstm_fcn-emg1k5_2 --use-max-stream --cudnn-lstm --save-best --save-all --log-to-file --verbose 2
python train.py mlstm_fcn emg9k_2 regression --epochs 250 --lr 0.001 --lr-reducer --early-stopping --name mlstm_fcn-emg9k_2 --use-max-stream --cudnn-lstm --save-best --save-all --log-to-file --verbose 2
python train.py mlstm_fcn sp5001k5_2 regression --epochs 250 --lr 0.001 --lr-reducer --early-stopping --name mlstm_fcn-sp5001k5_2 --use-max-stream --cudnn-lstm --save-best --save-all --log-to-file --verbose 2
python train.py mlstm_fcn sp5009k_2 regression --epochs 250 --lr 0.001 --lr-reducer --early-stopping --name mlstm_fcn-sp5009k_2 --use-max-stream --cudnn-lstm --save-best --save-all --log-to-file --verbose 2

echo "Training TCN referent model"
python train.py tcn lomax1k5_2 regression --epochs 250 --lr 0.001 --lr-reducer --early-stopping --name tcn-lomax1k5_2 --use-max-stream --cudnn-lstm --save-best --save-all --log-to-file --verbose 2
python train.py tcn lomax9k_2 regression --epochs 250 --lr 0.001 --lr-reducer --early-stopping --name tcn-lomax9k_2 --use-max-stream --cudnn-lstm --save-best --save-all --log-to-file --verbose 2
python train.py tcn lendb1k5_2 regression --epochs 250 --lr 0.001 --lr-reducer --early-stopping --name tcn-lendb1k5_2 --use-max-stream --cudnn-lstm --save-best --save-all --log-to-file --verbose 2
python train.py tcn lendb9k_2 regression --epochs 250 --lr 0.001 --lr-reducer --early-stopping --name tcn-lendb9k_2 --use-max-stream --cudnn-lstm --save-best --save-all --log-to-file --verbose 2
python train.py tcn stead1k5_2 regression --epochs 250 --lr 0.001 --lr-reducer --early-stopping --name tcn-stead1k5_2 --cudnn-lstm --save-best --save-all --log-to-file --verbose 2
python train.py tcn stead9k_2 regression --epochs 250 --lr 0.001 --lr-reducer --early-stopping --name tcn-stead9k_2 --cudnn-lstm --save-best --save-all --log-to-file --verbose 2

python train.py tcn speech8khz1k5_2 regression --epochs 250 --lr 0.001 --lr-reducer --early-stopping --name tcn-speech8khz1k5_2 --use-max-stream --cudnn-lstm --save-best --save-all --log-to-file --verbose 2
python train.py tcn speech8khz9k_2 regression --epochs 250 --lr 0.001 --lr-reducer --early-stopping --name tcn-speech8khz9k_2 --use-max-stream --cudnn-lstm --save-best --save-all --log-to-file --verbose 2
python train.py tcn emg1k5_2 regression --epochs 250 --lr 0.001 --lr-reducer --early-stopping --name tcn-emg1k5_2 --use-max-stream --cudnn-lstm --save-best --save-all --log-to-file --verbose 2
python train.py tcn emg9k_2 regression --epochs 250 --lr 0.001 --lr-reducer --early-stopping --name tcn-emg9k_2 --use-max-stream --cudnn-lstm --save-best --save-all --log-to-file --verbose 2
python train.py tcn sp5001k5_2 regression --epochs 250 --lr 0.001 --lr-reducer --early-stopping --name tcn-sp5001k5_2 --use-max-stream --cudnn-lstm --save-best --save-all --log-to-file --verbose 2
python train.py tcn sp5009k_2 regression --epochs 250 --lr 0.001 --lr-reducer --early-stopping --name tcn-sp5009k_2 --use-max-stream --cudnn-lstm --save-best --save-all --log-to-file --verbose 2

# 4th run
echo "Training ConvNetQuakeINGV referent model"
python train.py convnetquakeingv lomax1k5_3 regression --epochs 250 --lr 0.001 --lr-reducer --early-stopping --name convnetquakeingv-lomax1k5_3 --use-max-stream --cudnn-lstm --save-best --save-all --log-to-file --verbose 2
python train.py convnetquakeingv lomax9k_3 regression --epochs 250 --lr 0.001 --lr-reducer --early-stopping --name convnetquakeingv-lomax9k_3 --use-max-stream --cudnn-lstm --save-best --save-all --log-to-file --verbose 2
python train.py convnetquakeingv lendb1k5_3 regression --epochs 250 --lr 0.001 --lr-reducer --early-stopping --name convnetquakeingv-lendb1k5_3 --use-max-stream --cudnn-lstm --save-best --save-all --log-to-file --verbose 2
python train.py convnetquakeingv lendb9k_3 regression --epochs 250 --lr 0.001 --lr-reducer --early-stopping --name convnetquakeingv-lendb9k_3 --use-max-stream --cudnn-lstm --save-best --save-all --log-to-file --verbose 2
python train.py convnetquakeingv stead1k5_3 regression --epochs 250 --lr 0.001 --lr-reducer --early-stopping --name convnetquakeingv-stead1k5_3 --cudnn-lstm --save-best --save-all --log-to-file --verbose 2
python train.py convnetquakeingv stead9k_3 regression --epochs 250 --lr 0.001 --lr-reducer --early-stopping --name convnetquakeingv-stead9k_3 --cudnn-lstm --save-best --save-all --log-to-file --verbose 2

python train.py convnetquakeingv speech8khz1k5_3 regression --epochs 250 --lr 0.001 --lr-reducer --early-stopping --name convnetquakeingv-speech8khz1k5_3 --use-max-stream --cudnn-lstm --save-best --save-all --log-to-file --verbose 2
python train.py convnetquakeingv speech8khz9k_3 regression --epochs 250 --lr 0.001 --lr-reducer --early-stopping --name convnetquakeingv-speech8khz9k_3 --use-max-stream --cudnn-lstm --save-best --save-all --log-to-file --verbose 2
python train.py convnetquakeingv emg1k5_3 regression --epochs 250 --lr 0.001 --lr-reducer --early-stopping --name convnetquakeingv-emg1k5_3 --use-max-stream --cudnn-lstm --save-best --save-all --log-to-file --verbose 2
python train.py convnetquakeingv emg9k_3 regression --epochs 250 --lr 0.001 --lr-reducer --early-stopping --name convnetquakeingv-emg9k_3 --use-max-stream --cudnn-lstm --save-best --save-all --log-to-file --verbose 2
python train.py convnetquakeingv sp5001k5_3 regression --epochs 250 --lr 0.001 --lr-reducer --early-stopping --name convnetquakeingv-sp5001k5_3 --use-max-stream --cudnn-lstm --save-best --save-all --log-to-file --verbose 2
python train.py convnetquakeingv sp5009k_3 regression --epochs 250 --lr 0.001 --lr-reducer --early-stopping --name convnetquakeingv-sp5009k_3 --use-max-stream --cudnn-lstm --save-best --save-all --log-to-file --verbose 2

echo "Training MagNet referent model"
python train.py magnet lomax1k5_3 regression --epochs 250 --lr 0.001 --lr-reducer --early-stopping --name magnet-lomax1k5_3 --use-max-stream --cudnn-lstm --save-best --save-all --log-to-file --verbose 2
python train.py magnet lomax9k_3 regression --epochs 250 --lr 0.001 --lr-reducer --early-stopping --name magnet-lomax9k_3 --use-max-stream --cudnn-lstm --save-best --save-all --log-to-file --verbose 2
python train.py magnet lendb1k5_3 regression --epochs 250 --lr 0.001 --lr-reducer --early-stopping --name magnet-lendb1k5_3 --use-max-stream --cudnn-lstm --save-best --save-all --log-to-file --verbose 2
python train.py magnet lendb9k_3 regression --epochs 250 --lr 0.001 --lr-reducer --early-stopping --name magnet-lendb9k_3 --use-max-stream --cudnn-lstm --save-best --save-all --log-to-file --verbose 2
python train.py magnet stead1k5_3 regression --epochs 250 --lr 0.001 --lr-reducer --early-stopping --name magnet-stead1k5_3 --cudnn-lstm --save-best --save-all --log-to-file --verbose 2
python train.py magnet stead9k_3 regression --epochs 250 --lr 0.001 --lr-reducer --early-stopping --name magnet-stead9k_3 --cudnn-lstm --save-best --save-all --log-to-file --verbose 2

python train.py magnet speech8khz1k5_3 regression --epochs 250 --lr 0.001 --lr-reducer --early-stopping --name magnet-speech8khz1k5_3 --use-max-stream --cudnn-lstm --save-best --save-all --log-to-file --verbose 2
python train.py magnet speech8khz9k_3 regression --epochs 250 --lr 0.001 --lr-reducer --early-stopping --name magnet-speech8khz9k_3 --use-max-stream --cudnn-lstm --save-best --save-all --log-to-file --verbose 2
python train.py magnet emg1k5_3 regression --epochs 250 --lr 0.001 --lr-reducer --early-stopping --name magnet-emg1k5_3 --use-max-stream --cudnn-lstm --save-best --save-all --log-to-file --verbose 2
python train.py magnet emg9k_3 regression --epochs 250 --lr 0.001 --lr-reducer --early-stopping --name magnet-emg9k_3 --use-max-stream --cudnn-lstm --save-best --save-all --log-to-file --verbose 2
python train.py magnet sp5001k5_3 regression --epochs 250 --lr 0.001 --lr-reducer --early-stopping --name magnet-sp5001k5_3 --use-max-stream --cudnn-lstm --save-best --save-all --log-to-file --verbose 2
python train.py magnet sp5009k_3 regression --epochs 250 --lr 0.001 --lr-reducer --early-stopping --name magnet-sp5009k_3 --use-max-stream --cudnn-lstm --save-best --save-all --log-to-file --verbose 2

echo "Training MLSTM_FCN referent model"
python train.py mlstm_fcn lomax1k5_3 regression --epochs 250 --lr 0.001 --lr-reducer --early-stopping --name mlstm_fcn-lomax1k5_3 --use-max-stream --cudnn-lstm --save-best --save-all --log-to-file --verbose 2
python train.py mlstm_fcn lomax9k_3 regression --epochs 250 --lr 0.001 --lr-reducer --early-stopping --name mlstm_fcn-lomax9k_3 --use-max-stream --cudnn-lstm --save-best --save-all --log-to-file --verbose 2
python train.py mlstm_fcn lendb1k5_3 regression --epochs 250 --lr 0.001 --lr-reducer --early-stopping --name mlstm_fcn-lendb1k5_3 --use-max-stream --cudnn-lstm --save-best --save-all --log-to-file --verbose 2
python train.py mlstm_fcn lendb9k_3 regression --epochs 250 --lr 0.001 --lr-reducer --early-stopping --name mlstm_fcn-lendb9k_3 --use-max-stream --cudnn-lstm --save-best --save-all --log-to-file --verbose 2
python train.py mlstm_fcn stead1k5_3 regression --epochs 250 --lr 0.001 --lr-reducer --early-stopping --name mlstm_fcn-stead1k5_3 --cudnn-lstm --save-best --save-all --log-to-file --verbose 2
python train.py mlstm_fcn stead9k_3 regression --epochs 250 --lr 0.001 --lr-reducer --early-stopping --name mlstm_fcn-stead9k_3 --cudnn-lstm --save-best --save-all --log-to-file --verbose 2

python train.py mlstm_fcn speech8khz1k5_3 regression --epochs 250 --lr 0.001 --lr-reducer --early-stopping --name mlstm_fcn-speech8khz1k5_3 --use-max-stream --cudnn-lstm --save-best --save-all --log-to-file --verbose 2
python train.py mlstm_fcn speech8khz9k_3 regression --epochs 250 --lr 0.001 --lr-reducer --early-stopping --name mlstm_fcn-speech8khz9k_3 --use-max-stream --cudnn-lstm --save-best --save-all --log-to-file --verbose 2
python train.py mlstm_fcn emg1k5_3 regression --epochs 250 --lr 0.001 --lr-reducer --early-stopping --name mlstm_fcn-emg1k5_3 --use-max-stream --cudnn-lstm --save-best --save-all --log-to-file --verbose 2
python train.py mlstm_fcn emg9k_3 regression --epochs 250 --lr 0.001 --lr-reducer --early-stopping --name mlstm_fcn-emg9k_3 --use-max-stream --cudnn-lstm --save-best --save-all --log-to-file --verbose 2
python train.py mlstm_fcn sp5001k5_3 regression --epochs 250 --lr 0.001 --lr-reducer --early-stopping --name mlstm_fcn-sp5001k5_3 --use-max-stream --cudnn-lstm --save-best --save-all --log-to-file --verbose 2
python train.py mlstm_fcn sp5009k_3 regression --epochs 250 --lr 0.001 --lr-reducer --early-stopping --name mlstm_fcn-sp5009k_3 --use-max-stream --cudnn-lstm --save-best --save-all --log-to-file --verbose 2

echo "Training TCN referent model"
python train.py tcn lomax1k5_3 regression --epochs 250 --lr 0.001 --lr-reducer --early-stopping --name tcn-lomax1k5_3 --use-max-stream --cudnn-lstm --save-best --save-all --log-to-file --verbose 2
python train.py tcn lomax9k_3 regression --epochs 250 --lr 0.001 --lr-reducer --early-stopping --name tcn-lomax9k_3 --use-max-stream --cudnn-lstm --save-best --save-all --log-to-file --verbose 2
python train.py tcn lendb1k5_3 regression --epochs 250 --lr 0.001 --lr-reducer --early-stopping --name tcn-lendb1k5_3 --use-max-stream --cudnn-lstm --save-best --save-all --log-to-file --verbose 2
python train.py tcn lendb9k_3 regression --epochs 250 --lr 0.001 --lr-reducer --early-stopping --name tcn-lendb9k_3 --use-max-stream --cudnn-lstm --save-best --save-all --log-to-file --verbose 2
python train.py tcn stead1k5_3 regression --epochs 250 --lr 0.001 --lr-reducer --early-stopping --name tcn-stead1k5_3 --cudnn-lstm --save-best --save-all --log-to-file --verbose 2
python train.py tcn stead9k_3 regression --epochs 250 --lr 0.001 --lr-reducer --early-stopping --name tcn-stead9k_3 --cudnn-lstm --save-best --save-all --log-to-file --verbose 2

python train.py tcn speech8khz1k5_3 regression --epochs 250 --lr 0.001 --lr-reducer --early-stopping --name tcn-speech8khz1k5_3 --use-max-stream --cudnn-lstm --save-best --save-all --log-to-file --verbose 2
python train.py tcn speech8khz9k_3 regression --epochs 250 --lr 0.001 --lr-reducer --early-stopping --name tcn-speech8khz9k_3 --use-max-stream --cudnn-lstm --save-best --save-all --log-to-file --verbose 2
python train.py tcn emg1k5_3 regression --epochs 250 --lr 0.001 --lr-reducer --early-stopping --name tcn-emg1k5_3 --use-max-stream --cudnn-lstm --save-best --save-all --log-to-file --verbose 2
python train.py tcn emg9k_3 regression --epochs 250 --lr 0.001 --lr-reducer --early-stopping --name tcn-emg9k_3 --use-max-stream --cudnn-lstm --save-best --save-all --log-to-file --verbose 2
python train.py tcn sp5001k5_3 regression --epochs 250 --lr 0.001 --lr-reducer --early-stopping --name tcn-sp5001k5_3 --use-max-stream --cudnn-lstm --save-best --save-all --log-to-file --verbose 2
python train.py tcn sp5009k_3 regression --epochs 250 --lr 0.001 --lr-reducer --early-stopping --name tcn-sp5009k_3 --use-max-stream --cudnn-lstm --save-best --save-all --log-to-file --verbose 2

# 5th run
echo "Training ConvNetQuakeINGV referent model"
python train.py convnetquakeingv lomax1k5_4 regression --epochs 250 --lr 0.001 --lr-reducer --early-stopping --name convnetquakeingv-lomax1k5_4 --use-max-stream --cudnn-lstm --save-best --save-all --log-to-file --verbose 2
python train.py convnetquakeingv lomax9k_4 regression --epochs 250 --lr 0.001 --lr-reducer --early-stopping --name convnetquakeingv-lomax9k_4 --use-max-stream --cudnn-lstm --save-best --save-all --log-to-file --verbose 2
python train.py convnetquakeingv lendb1k5_4 regression --epochs 250 --lr 0.001 --lr-reducer --early-stopping --name convnetquakeingv-lendb1k5_4 --use-max-stream --cudnn-lstm --save-best --save-all --log-to-file --verbose 2
python train.py convnetquakeingv lendb9k_4 regression --epochs 250 --lr 0.001 --lr-reducer --early-stopping --name convnetquakeingv-lendb9k_4 --use-max-stream --cudnn-lstm --save-best --save-all --log-to-file --verbose 2
python train.py convnetquakeingv stead1k5_4 regression --epochs 250 --lr 0.001 --lr-reducer --early-stopping --name convnetquakeingv-stead1k5_4 --cudnn-lstm --save-best --save-all --log-to-file --verbose 2
python train.py convnetquakeingv stead9k_4 regression --epochs 250 --lr 0.001 --lr-reducer --early-stopping --name convnetquakeingv-stead9k_4 --cudnn-lstm --save-best --save-all --log-to-file --verbose 2

python train.py convnetquakeingv speech8khz1k5_4 regression --epochs 250 --lr 0.001 --lr-reducer --early-stopping --name convnetquakeingv-speech8khz1k5_4 --use-max-stream --cudnn-lstm --save-best --save-all --log-to-file --verbose 2
python train.py convnetquakeingv speech8khz9k_4 regression --epochs 250 --lr 0.001 --lr-reducer --early-stopping --name convnetquakeingv-speech8khz9k_4 --use-max-stream --cudnn-lstm --save-best --save-all --log-to-file --verbose 2
python train.py convnetquakeingv emg1k5_4 regression --epochs 250 --lr 0.001 --lr-reducer --early-stopping --name convnetquakeingv-emg1k5_4 --use-max-stream --cudnn-lstm --save-best --save-all --log-to-file --verbose 2
python train.py convnetquakeingv emg9k_4 regression --epochs 250 --lr 0.001 --lr-reducer --early-stopping --name convnetquakeingv-emg9k_4 --use-max-stream --cudnn-lstm --save-best --save-all --log-to-file --verbose 2
python train.py convnetquakeingv sp5001k5_4 regression --epochs 250 --lr 0.001 --lr-reducer --early-stopping --name convnetquakeingv-sp5001k5_4 --use-max-stream --cudnn-lstm --save-best --save-all --log-to-file --verbose 2
python train.py convnetquakeingv sp5009k_4 regression --epochs 250 --lr 0.001 --lr-reducer --early-stopping --name convnetquakeingv-sp5009k_4 --use-max-stream --cudnn-lstm --save-best --save-all --log-to-file --verbose 2

echo "Training MagNet referent model"
python train.py magnet lomax1k5_4 regression --epochs 250 --lr 0.001 --lr-reducer --early-stopping --name magnet-lomax1k5_4 --use-max-stream --cudnn-lstm --save-best --save-all --log-to-file --verbose 2
python train.py magnet lomax9k_4 regression --epochs 250 --lr 0.001 --lr-reducer --early-stopping --name magnet-lomax9k_4 --use-max-stream --cudnn-lstm --save-best --save-all --log-to-file --verbose 2
python train.py magnet lendb1k5_4 regression --epochs 250 --lr 0.001 --lr-reducer --early-stopping --name magnet-lendb1k5_4 --use-max-stream --cudnn-lstm --save-best --save-all --log-to-file --verbose 2
python train.py magnet lendb9k_4 regression --epochs 250 --lr 0.001 --lr-reducer --early-stopping --name magnet-lendb9k_4 --use-max-stream --cudnn-lstm --save-best --save-all --log-to-file --verbose 2
python train.py magnet stead1k5_4 regression --epochs 250 --lr 0.001 --lr-reducer --early-stopping --name magnet-stead1k5_4 --cudnn-lstm --save-best --save-all --log-to-file --verbose 2
python train.py magnet stead9k_4 regression --epochs 250 --lr 0.001 --lr-reducer --early-stopping --name magnet-stead9k_4 --cudnn-lstm --save-best --save-all --log-to-file --verbose 2

python train.py magnet speech8khz1k5_4 regression --epochs 250 --lr 0.001 --lr-reducer --early-stopping --name magnet-speech8khz1k5_4 --use-max-stream --cudnn-lstm --save-best --save-all --log-to-file --verbose 2
python train.py magnet speech8khz9k_4 regression --epochs 250 --lr 0.001 --lr-reducer --early-stopping --name magnet-speech8khz9k_4 --use-max-stream --cudnn-lstm --save-best --save-all --log-to-file --verbose 2
python train.py magnet emg1k5_4 regression --epochs 250 --lr 0.001 --lr-reducer --early-stopping --name magnet-emg1k5_4 --use-max-stream --cudnn-lstm --save-best --save-all --log-to-file --verbose 2
python train.py magnet emg9k_4 regression --epochs 250 --lr 0.001 --lr-reducer --early-stopping --name magnet-emg9k_4 --use-max-stream --cudnn-lstm --save-best --save-all --log-to-file --verbose 2
python train.py magnet sp5001k5_4 regression --epochs 250 --lr 0.001 --lr-reducer --early-stopping --name magnet-sp5001k5_4 --use-max-stream --cudnn-lstm --save-best --save-all --log-to-file --verbose 2
python train.py magnet sp5009k_4 regression --epochs 250 --lr 0.001 --lr-reducer --early-stopping --name magnet-sp5009k_4 --use-max-stream --cudnn-lstm --save-best --save-all --log-to-file --verbose 2

echo "Training MLSTM_FCN referent model"
python train.py mlstm_fcn lomax1k5_4 regression --epochs 250 --lr 0.001 --lr-reducer --early-stopping --name mlstm_fcn-lomax1k5_4 --use-max-stream --cudnn-lstm --save-best --save-all --log-to-file --verbose 2
python train.py mlstm_fcn lomax9k_4 regression --epochs 250 --lr 0.001 --lr-reducer --early-stopping --name mlstm_fcn-lomax9k_4 --use-max-stream --cudnn-lstm --save-best --save-all --log-to-file --verbose 2
python train.py mlstm_fcn lendb1k5_4 regression --epochs 250 --lr 0.001 --lr-reducer --early-stopping --name mlstm_fcn-lendb1k5_4 --use-max-stream --cudnn-lstm --save-best --save-all --log-to-file --verbose 2
python train.py mlstm_fcn lendb9k_4 regression --epochs 250 --lr 0.001 --lr-reducer --early-stopping --name mlstm_fcn-lendb9k_4 --use-max-stream --cudnn-lstm --save-best --save-all --log-to-file --verbose 2
python train.py mlstm_fcn stead1k5_4 regression --epochs 250 --lr 0.001 --lr-reducer --early-stopping --name mlstm_fcn-stead1k5_4 --cudnn-lstm --save-best --save-all --log-to-file --verbose 2
python train.py mlstm_fcn stead9k_4 regression --epochs 250 --lr 0.001 --lr-reducer --early-stopping --name mlstm_fcn-stead9k_4 --cudnn-lstm --save-best --save-all --log-to-file --verbose 2

python train.py mlstm_fcn speech8khz1k5_4 regression --epochs 250 --lr 0.001 --lr-reducer --early-stopping --name mlstm_fcn-speech8khz1k5_4 --use-max-stream --cudnn-lstm --save-best --save-all --log-to-file --verbose 2
python train.py mlstm_fcn speech8khz9k_4 regression --epochs 250 --lr 0.001 --lr-reducer --early-stopping --name mlstm_fcn-speech8khz9k_4 --use-max-stream --cudnn-lstm --save-best --save-all --log-to-file --verbose 2
python train.py mlstm_fcn emg1k5_4 regression --epochs 250 --lr 0.001 --lr-reducer --early-stopping --name mlstm_fcn-emg1k5_4 --use-max-stream --cudnn-lstm --save-best --save-all --log-to-file --verbose 2
python train.py mlstm_fcn emg9k_4 regression --epochs 250 --lr 0.001 --lr-reducer --early-stopping --name mlstm_fcn-emg9k_4 --use-max-stream --cudnn-lstm --save-best --save-all --log-to-file --verbose 2
python train.py mlstm_fcn sp5001k5_4 regression --epochs 250 --lr 0.001 --lr-reducer --early-stopping --name mlstm_fcn-sp5001k5_4 --use-max-stream --cudnn-lstm --save-best --save-all --log-to-file --verbose 2
python train.py mlstm_fcn sp5009k_4 regression --epochs 250 --lr 0.001 --lr-reducer --early-stopping --name mlstm_fcn-sp5009k_4 --use-max-stream --cudnn-lstm --save-best --save-all --log-to-file --verbose 2

echo "Training TCN referent model"
python train.py tcn lomax1k5_4 regression --epochs 250 --lr 0.001 --lr-reducer --early-stopping --name tcn-lomax1k5_4 --use-max-stream --cudnn-lstm --save-best --save-all --log-to-file --verbose 2
python train.py tcn lomax9k_4 regression --epochs 250 --lr 0.001 --lr-reducer --early-stopping --name tcn-lomax9k_4 --use-max-stream --cudnn-lstm --save-best --save-all --log-to-file --verbose 2
python train.py tcn lendb1k5_4 regression --epochs 250 --lr 0.001 --lr-reducer --early-stopping --name tcn-lendb1k5_4 --use-max-stream --cudnn-lstm --save-best --save-all --log-to-file --verbose 2
python train.py tcn lendb9k_4 regression --epochs 250 --lr 0.001 --lr-reducer --early-stopping --name tcn-lendb9k_4 --use-max-stream --cudnn-lstm --save-best --save-all --log-to-file --verbose 2
python train.py tcn stead1k5_4 regression --epochs 250 --lr 0.001 --lr-reducer --early-stopping --name tcn-stead1k5_4 --cudnn-lstm --save-best --save-all --log-to-file --verbose 2
python train.py tcn stead9k_4 regression --epochs 250 --lr 0.001 --lr-reducer --early-stopping --name tcn-stead9k_4 --cudnn-lstm --save-best --save-all --log-to-file --verbose 2

python train.py tcn speech8khz1k5_4 regression --epochs 250 --lr 0.001 --lr-reducer --early-stopping --name tcn-speech8khz1k5_4 --use-max-stream --cudnn-lstm --save-best --save-all --log-to-file --verbose 2
python train.py tcn speech8khz9k_4 regression --epochs 250 --lr 0.001 --lr-reducer --early-stopping --name tcn-speech8khz9k_4 --use-max-stream --cudnn-lstm --save-best --save-all --log-to-file --verbose 2
python train.py tcn emg1k5_4 regression --epochs 250 --lr 0.001 --lr-reducer --early-stopping --name tcn-emg1k5_4 --use-max-stream --cudnn-lstm --save-best --save-all --log-to-file --verbose 2
python train.py tcn emg9k_4 regression --epochs 250 --lr 0.001 --lr-reducer --early-stopping --name tcn-emg9k_4 --use-max-stream --cudnn-lstm --save-best --save-all --log-to-file --verbose 2
python train.py tcn sp5001k5_4 regression --epochs 250 --lr 0.001 --lr-reducer --early-stopping --name tcn-sp5001k5_4 --use-max-stream --cudnn-lstm --save-best --save-all --log-to-file --verbose 2
python train.py tcn sp5009k_4 regression --epochs 250 --lr 0.001 --lr-reducer --early-stopping --name tcn-sp5009k_4 --use-max-stream --cudnn-lstm --save-best --save-all --log-to-file --verbose 2

# 6th run
echo "Training ConvNetQuakeINGV referent model"
python train.py convnetquakeingv lomax1k5_5 regression --epochs 250 --lr 0.001 --lr-reducer --early-stopping --name convnetquakeingv-lomax1k5_5 --use-max-stream --cudnn-lstm --save-best --save-all --log-to-file --verbose 2
python train.py convnetquakeingv lomax9k_5 regression --epochs 250 --lr 0.001 --lr-reducer --early-stopping --name convnetquakeingv-lomax9k_5 --use-max-stream --cudnn-lstm --save-best --save-all --log-to-file --verbose 2
python train.py convnetquakeingv lendb1k5_5 regression --epochs 250 --lr 0.001 --lr-reducer --early-stopping --name convnetquakeingv-lendb1k5_5 --use-max-stream --cudnn-lstm --save-best --save-all --log-to-file --verbose 2
python train.py convnetquakeingv lendb9k_5 regression --epochs 250 --lr 0.001 --lr-reducer --early-stopping --name convnetquakeingv-lendb9k_5 --use-max-stream --cudnn-lstm --save-best --save-all --log-to-file --verbose 2
python train.py convnetquakeingv stead1k5_5 regression --epochs 250 --lr 0.001 --lr-reducer --early-stopping --name convnetquakeingv-stead1k5_5 --cudnn-lstm --save-best --save-all --log-to-file --verbose 2
python train.py convnetquakeingv stead9k_5 regression --epochs 250 --lr 0.001 --lr-reducer --early-stopping --name convnetquakeingv-stead9k_5 --cudnn-lstm --save-best --save-all --log-to-file --verbose 2

python train.py convnetquakeingv speech8khz1k5_5 regression --epochs 250 --lr 0.001 --lr-reducer --early-stopping --name convnetquakeingv-speech8khz1k5_5 --use-max-stream --cudnn-lstm --save-best --save-all --log-to-file --verbose 2
python train.py convnetquakeingv speech8khz9k_5 regression --epochs 250 --lr 0.001 --lr-reducer --early-stopping --name convnetquakeingv-speech8khz9k_5 --use-max-stream --cudnn-lstm --save-best --save-all --log-to-file --verbose 2
python train.py convnetquakeingv emg1k5_5 regression --epochs 250 --lr 0.001 --lr-reducer --early-stopping --name convnetquakeingv-emg1k5_5 --use-max-stream --cudnn-lstm --save-best --save-all --log-to-file --verbose 2
python train.py convnetquakeingv emg9k_5 regression --epochs 250 --lr 0.001 --lr-reducer --early-stopping --name convnetquakeingv-emg9k_5 --use-max-stream --cudnn-lstm --save-best --save-all --log-to-file --verbose 2
python train.py convnetquakeingv sp5001k5_5 regression --epochs 250 --lr 0.001 --lr-reducer --early-stopping --name convnetquakeingv-sp5001k5_5 --use-max-stream --cudnn-lstm --save-best --save-all --log-to-file --verbose 2
python train.py convnetquakeingv sp5009k_5 regression --epochs 250 --lr 0.001 --lr-reducer --early-stopping --name convnetquakeingv-sp5009k_5 --use-max-stream --cudnn-lstm --save-best --save-all --log-to-file --verbose 2

echo "Training MagNet referent model"
python train.py magnet lomax1k5_5 regression --epochs 250 --lr 0.001 --lr-reducer --early-stopping --name magnet-lomax1k5_5 --use-max-stream --cudnn-lstm --save-best --save-all --log-to-file --verbose 2
python train.py magnet lomax9k_5 regression --epochs 250 --lr 0.001 --lr-reducer --early-stopping --name magnet-lomax9k_5 --use-max-stream --cudnn-lstm --save-best --save-all --log-to-file --verbose 2
python train.py magnet lendb1k5_5 regression --epochs 250 --lr 0.001 --lr-reducer --early-stopping --name magnet-lendb1k5_5 --use-max-stream --cudnn-lstm --save-best --save-all --log-to-file --verbose 2
python train.py magnet lendb9k_5 regression --epochs 250 --lr 0.001 --lr-reducer --early-stopping --name magnet-lendb9k_5 --use-max-stream --cudnn-lstm --save-best --save-all --log-to-file --verbose 2
python train.py magnet stead1k5_5 regression --epochs 250 --lr 0.001 --lr-reducer --early-stopping --name magnet-stead1k5_5 --cudnn-lstm --save-best --save-all --log-to-file --verbose 2
python train.py magnet stead9k_5 regression --epochs 250 --lr 0.001 --lr-reducer --early-stopping --name magnet-stead9k_5 --cudnn-lstm --save-best --save-all --log-to-file --verbose 2

python train.py magnet speech8khz1k5_5 regression --epochs 250 --lr 0.001 --lr-reducer --early-stopping --name magnet-speech8khz1k5_5 --use-max-stream --cudnn-lstm --save-best --save-all --log-to-file --verbose 2
python train.py magnet speech8khz9k_5 regression --epochs 250 --lr 0.001 --lr-reducer --early-stopping --name magnet-speech8khz9k_5 --use-max-stream --cudnn-lstm --save-best --save-all --log-to-file --verbose 2
python train.py magnet emg1k5_5 regression --epochs 250 --lr 0.001 --lr-reducer --early-stopping --name magnet-emg1k5_5 --use-max-stream --cudnn-lstm --save-best --save-all --log-to-file --verbose 2
python train.py magnet emg9k_5 regression --epochs 250 --lr 0.001 --lr-reducer --early-stopping --name magnet-emg9k_5 --use-max-stream --cudnn-lstm --save-best --save-all --log-to-file --verbose 2
python train.py magnet sp5001k5_5 regression --epochs 250 --lr 0.001 --lr-reducer --early-stopping --name magnet-sp5001k5_5 --use-max-stream --cudnn-lstm --save-best --save-all --log-to-file --verbose 2
python train.py magnet sp5009k_5 regression --epochs 250 --lr 0.001 --lr-reducer --early-stopping --name magnet-sp5009k_5 --use-max-stream --cudnn-lstm --save-best --save-all --log-to-file --verbose 2

echo "Training MLSTM_FCN referent model"
python train.py mlstm_fcn lomax1k5_5 regression --epochs 250 --lr 0.001 --lr-reducer --early-stopping --name mlstm_fcn-lomax1k5_5 --use-max-stream --cudnn-lstm --save-best --save-all --log-to-file --verbose 2
python train.py mlstm_fcn lomax9k_5 regression --epochs 250 --lr 0.001 --lr-reducer --early-stopping --name mlstm_fcn-lomax9k_5 --use-max-stream --cudnn-lstm --save-best --save-all --log-to-file --verbose 2
python train.py mlstm_fcn lendb1k5_5 regression --epochs 250 --lr 0.001 --lr-reducer --early-stopping --name mlstm_fcn-lendb1k5_5 --use-max-stream --cudnn-lstm --save-best --save-all --log-to-file --verbose 2
python train.py mlstm_fcn lendb9k_5 regression --epochs 250 --lr 0.001 --lr-reducer --early-stopping --name mlstm_fcn-lendb9k_5 --use-max-stream --cudnn-lstm --save-best --save-all --log-to-file --verbose 2
python train.py mlstm_fcn stead1k5_5 regression --epochs 250 --lr 0.001 --lr-reducer --early-stopping --name mlstm_fcn-stead1k5_5 --cudnn-lstm --save-best --save-all --log-to-file --verbose 2
python train.py mlstm_fcn stead9k_5 regression --epochs 250 --lr 0.001 --lr-reducer --early-stopping --name mlstm_fcn-stead9k_5 --cudnn-lstm --save-best --save-all --log-to-file --verbose 2

python train.py mlstm_fcn speech8khz1k5_5 regression --epochs 250 --lr 0.001 --lr-reducer --early-stopping --name mlstm_fcn-speech8khz1k5_5 --use-max-stream --cudnn-lstm --save-best --save-all --log-to-file --verbose 2
python train.py mlstm_fcn speech8khz9k_5 regression --epochs 250 --lr 0.001 --lr-reducer --early-stopping --name mlstm_fcn-speech8khz9k_5 --use-max-stream --cudnn-lstm --save-best --save-all --log-to-file --verbose 2
python train.py mlstm_fcn emg1k5_5 regression --epochs 250 --lr 0.001 --lr-reducer --early-stopping --name mlstm_fcn-emg1k5_5 --use-max-stream --cudnn-lstm --save-best --save-all --log-to-file --verbose 2
python train.py mlstm_fcn emg9k_5 regression --epochs 250 --lr 0.001 --lr-reducer --early-stopping --name mlstm_fcn-emg9k_5 --use-max-stream --cudnn-lstm --save-best --save-all --log-to-file --verbose 2
python train.py mlstm_fcn sp5001k5_5 regression --epochs 250 --lr 0.001 --lr-reducer --early-stopping --name mlstm_fcn-sp5001k5_5 --use-max-stream --cudnn-lstm --save-best --save-all --log-to-file --verbose 2
python train.py mlstm_fcn sp5009k_5 regression --epochs 250 --lr 0.001 --lr-reducer --early-stopping --name mlstm_fcn-sp5009k_5 --use-max-stream --cudnn-lstm --save-best --save-all --log-to-file --verbose 2

echo "Training TCN referent model"
python train.py tcn lomax1k5_5 regression --epochs 250 --lr 0.001 --lr-reducer --early-stopping --name tcn-lomax1k5_5 --use-max-stream --cudnn-lstm --save-best --save-all --log-to-file --verbose 2
python train.py tcn lomax9k_5 regression --epochs 250 --lr 0.001 --lr-reducer --early-stopping --name tcn-lomax9k_5 --use-max-stream --cudnn-lstm --save-best --save-all --log-to-file --verbose 2
python train.py tcn lendb1k5_5 regression --epochs 250 --lr 0.001 --lr-reducer --early-stopping --name tcn-lendb1k5_5 --use-max-stream --cudnn-lstm --save-best --save-all --log-to-file --verbose 2
python train.py tcn lendb9k_5 regression --epochs 250 --lr 0.001 --lr-reducer --early-stopping --name tcn-lendb9k_5 --use-max-stream --cudnn-lstm --save-best --save-all --log-to-file --verbose 2
python train.py tcn stead1k5_5 regression --epochs 250 --lr 0.001 --lr-reducer --early-stopping --name tcn-stead1k5_5 --cudnn-lstm --save-best --save-all --log-to-file --verbose 2
python train.py tcn stead9k_5 regression --epochs 250 --lr 0.001 --lr-reducer --early-stopping --name tcn-stead9k_5 --cudnn-lstm --save-best --save-all --log-to-file --verbose 2

python train.py tcn speech8khz1k5_5 regression --epochs 250 --lr 0.001 --lr-reducer --early-stopping --name tcn-speech8khz1k5_5 --use-max-stream --cudnn-lstm --save-best --save-all --log-to-file --verbose 2
python train.py tcn speech8khz9k_5 regression --epochs 250 --lr 0.001 --lr-reducer --early-stopping --name tcn-speech8khz9k_5 --use-max-stream --cudnn-lstm --save-best --save-all --log-to-file --verbose 2
python train.py tcn emg1k5_5 regression --epochs 250 --lr 0.001 --lr-reducer --early-stopping --name tcn-emg1k5_5 --use-max-stream --cudnn-lstm --save-best --save-all --log-to-file --verbose 2
python train.py tcn emg9k_5 regression --epochs 250 --lr 0.001 --lr-reducer --early-stopping --name tcn-emg9k_5 --use-max-stream --cudnn-lstm --save-best --save-all --log-to-file --verbose 2
python train.py tcn sp5001k5_5 regression --epochs 250 --lr 0.001 --lr-reducer --early-stopping --name tcn-sp5001k5_5 --use-max-stream --cudnn-lstm --save-best --save-all --log-to-file --verbose 2
python train.py tcn sp5009k_5 regression --epochs 250 --lr 0.001 --lr-reducer --early-stopping --name tcn-sp5009k_5 --use-max-stream --cudnn-lstm --save-best --save-all --log-to-file --verbose 2

# 7th run
echo "Training ConvNetQuakeINGV referent model"
python train.py convnetquakeingv lomax1k5_6 regression --epochs 250 --lr 0.001 --lr-reducer --early-stopping --name convnetquakeingv-lomax1k5_6 --use-max-stream --cudnn-lstm --save-best --save-all --log-to-file --verbose 2
python train.py convnetquakeingv lomax9k_6 regression --epochs 250 --lr 0.001 --lr-reducer --early-stopping --name convnetquakeingv-lomax9k_6 --use-max-stream --cudnn-lstm --save-best --save-all --log-to-file --verbose 2
python train.py convnetquakeingv lendb1k5_6 regression --epochs 250 --lr 0.001 --lr-reducer --early-stopping --name convnetquakeingv-lendb1k5_6 --use-max-stream --cudnn-lstm --save-best --save-all --log-to-file --verbose 2
python train.py convnetquakeingv lendb9k_6 regression --epochs 250 --lr 0.001 --lr-reducer --early-stopping --name convnetquakeingv-lendb9k_6 --use-max-stream --cudnn-lstm --save-best --save-all --log-to-file --verbose 2
python train.py convnetquakeingv stead1k5_6 regression --epochs 250 --lr 0.001 --lr-reducer --early-stopping --name convnetquakeingv-stead1k5_6 --cudnn-lstm --save-best --save-all --log-to-file --verbose 2
python train.py convnetquakeingv stead9k_6 regression --epochs 250 --lr 0.001 --lr-reducer --early-stopping --name convnetquakeingv-stead9k_6 --cudnn-lstm --save-best --save-all --log-to-file --verbose 2

python train.py convnetquakeingv speech8khz1k5_6 regression --epochs 250 --lr 0.001 --lr-reducer --early-stopping --name convnetquakeingv-speech8khz1k5_6 --use-max-stream --cudnn-lstm --save-best --save-all --log-to-file --verbose 2
python train.py convnetquakeingv speech8khz9k_6 regression --epochs 250 --lr 0.001 --lr-reducer --early-stopping --name convnetquakeingv-speech8khz9k_6 --use-max-stream --cudnn-lstm --save-best --save-all --log-to-file --verbose 2
python train.py convnetquakeingv emg1k5_6 regression --epochs 250 --lr 0.001 --lr-reducer --early-stopping --name convnetquakeingv-emg1k5_6 --use-max-stream --cudnn-lstm --save-best --save-all --log-to-file --verbose 2
python train.py convnetquakeingv emg9k_6 regression --epochs 250 --lr 0.001 --lr-reducer --early-stopping --name convnetquakeingv-emg9k_6 --use-max-stream --cudnn-lstm --save-best --save-all --log-to-file --verbose 2
python train.py convnetquakeingv sp5001k5_6 regression --epochs 250 --lr 0.001 --lr-reducer --early-stopping --name convnetquakeingv-sp5001k5_6 --use-max-stream --cudnn-lstm --save-best --save-all --log-to-file --verbose 2
python train.py convnetquakeingv sp5009k_6 regression --epochs 250 --lr 0.001 --lr-reducer --early-stopping --name convnetquakeingv-sp5009k_6 --use-max-stream --cudnn-lstm --save-best --save-all --log-to-file --verbose 2

echo "Training MagNet referent model"
python train.py magnet lomax1k5_6 regression --epochs 250 --lr 0.001 --lr-reducer --early-stopping --name magnet-lomax1k5_6 --use-max-stream --cudnn-lstm --save-best --save-all --log-to-file --verbose 2
python train.py magnet lomax9k_6 regression --epochs 250 --lr 0.001 --lr-reducer --early-stopping --name magnet-lomax9k_6 --use-max-stream --cudnn-lstm --save-best --save-all --log-to-file --verbose 2
python train.py magnet lendb1k5_6 regression --epochs 250 --lr 0.001 --lr-reducer --early-stopping --name magnet-lendb1k5_6 --use-max-stream --cudnn-lstm --save-best --save-all --log-to-file --verbose 2
python train.py magnet lendb9k_6 regression --epochs 250 --lr 0.001 --lr-reducer --early-stopping --name magnet-lendb9k_6 --use-max-stream --cudnn-lstm --save-best --save-all --log-to-file --verbose 2
python train.py magnet stead1k5_6 regression --epochs 250 --lr 0.001 --lr-reducer --early-stopping --name magnet-stead1k5_6 --cudnn-lstm --save-best --save-all --log-to-file --verbose 2
python train.py magnet stead9k_6 regression --epochs 250 --lr 0.001 --lr-reducer --early-stopping --name magnet-stead9k_6 --cudnn-lstm --save-best --save-all --log-to-file --verbose 2

python train.py magnet speech8khz1k5_6 regression --epochs 250 --lr 0.001 --lr-reducer --early-stopping --name magnet-speech8khz1k5_6 --use-max-stream --cudnn-lstm --save-best --save-all --log-to-file --verbose 2
python train.py magnet speech8khz9k_6 regression --epochs 250 --lr 0.001 --lr-reducer --early-stopping --name magnet-speech8khz9k_6 --use-max-stream --cudnn-lstm --save-best --save-all --log-to-file --verbose 2
python train.py magnet emg1k5_6 regression --epochs 250 --lr 0.001 --lr-reducer --early-stopping --name magnet-emg1k5_6 --use-max-stream --cudnn-lstm --save-best --save-all --log-to-file --verbose 2
python train.py magnet emg9k_6 regression --epochs 250 --lr 0.001 --lr-reducer --early-stopping --name magnet-emg9k_6 --use-max-stream --cudnn-lstm --save-best --save-all --log-to-file --verbose 2
python train.py magnet sp5001k5_6 regression --epochs 250 --lr 0.001 --lr-reducer --early-stopping --name magnet-sp5001k5_6 --use-max-stream --cudnn-lstm --save-best --save-all --log-to-file --verbose 2
python train.py magnet sp5009k_6 regression --epochs 250 --lr 0.001 --lr-reducer --early-stopping --name magnet-sp5009k_6 --use-max-stream --cudnn-lstm --save-best --save-all --log-to-file --verbose 2

echo "Training MLSTM_FCN referent model"
python train.py mlstm_fcn lomax1k5_6 regression --epochs 250 --lr 0.001 --lr-reducer --early-stopping --name mlstm_fcn-lomax1k5_6 --use-max-stream --cudnn-lstm --save-best --save-all --log-to-file --verbose 2
python train.py mlstm_fcn lomax9k_6 regression --epochs 250 --lr 0.001 --lr-reducer --early-stopping --name mlstm_fcn-lomax9k_6 --use-max-stream --cudnn-lstm --save-best --save-all --log-to-file --verbose 2
python train.py mlstm_fcn lendb1k5_6 regression --epochs 250 --lr 0.001 --lr-reducer --early-stopping --name mlstm_fcn-lendb1k5_6 --use-max-stream --cudnn-lstm --save-best --save-all --log-to-file --verbose 2
python train.py mlstm_fcn lendb9k_6 regression --epochs 250 --lr 0.001 --lr-reducer --early-stopping --name mlstm_fcn-lendb9k_6 --use-max-stream --cudnn-lstm --save-best --save-all --log-to-file --verbose 2
python train.py mlstm_fcn stead1k5_6 regression --epochs 250 --lr 0.001 --lr-reducer --early-stopping --name mlstm_fcn-stead1k5_6 --cudnn-lstm --save-best --save-all --log-to-file --verbose 2
python train.py mlstm_fcn stead9k_6 regression --epochs 250 --lr 0.001 --lr-reducer --early-stopping --name mlstm_fcn-stead9k_6 --cudnn-lstm --save-best --save-all --log-to-file --verbose 2

python train.py mlstm_fcn speech8khz1k5_6 regression --epochs 250 --lr 0.001 --lr-reducer --early-stopping --name mlstm_fcn-speech8khz1k5_6 --use-max-stream --cudnn-lstm --save-best --save-all --log-to-file --verbose 2
python train.py mlstm_fcn speech8khz9k_6 regression --epochs 250 --lr 0.001 --lr-reducer --early-stopping --name mlstm_fcn-speech8khz9k_6 --use-max-stream --cudnn-lstm --save-best --save-all --log-to-file --verbose 2
python train.py mlstm_fcn emg1k5_6 regression --epochs 250 --lr 0.001 --lr-reducer --early-stopping --name mlstm_fcn-emg1k5_6 --use-max-stream --cudnn-lstm --save-best --save-all --log-to-file --verbose 2
python train.py mlstm_fcn emg9k_6 regression --epochs 250 --lr 0.001 --lr-reducer --early-stopping --name mlstm_fcn-emg9k_6 --use-max-stream --cudnn-lstm --save-best --save-all --log-to-file --verbose 2
python train.py mlstm_fcn sp5001k5_6 regression --epochs 250 --lr 0.001 --lr-reducer --early-stopping --name mlstm_fcn-sp5001k5_6 --use-max-stream --cudnn-lstm --save-best --save-all --log-to-file --verbose 2
python train.py mlstm_fcn sp5009k_6 regression --epochs 250 --lr 0.001 --lr-reducer --early-stopping --name mlstm_fcn-sp5009k_6 --use-max-stream --cudnn-lstm --save-best --save-all --log-to-file --verbose 2

echo "Training TCN referent model"
python train.py tcn lomax1k5_6 regression --epochs 250 --lr 0.001 --lr-reducer --early-stopping --name tcn-lomax1k5_6 --use-max-stream --cudnn-lstm --save-best --save-all --log-to-file --verbose 2
python train.py tcn lomax9k_6 regression --epochs 250 --lr 0.001 --lr-reducer --early-stopping --name tcn-lomax9k_6 --use-max-stream --cudnn-lstm --save-best --save-all --log-to-file --verbose 2
python train.py tcn lendb1k5_6 regression --epochs 250 --lr 0.001 --lr-reducer --early-stopping --name tcn-lendb1k5_6 --use-max-stream --cudnn-lstm --save-best --save-all --log-to-file --verbose 2
python train.py tcn lendb9k_6 regression --epochs 250 --lr 0.001 --lr-reducer --early-stopping --name tcn-lendb9k_6 --use-max-stream --cudnn-lstm --save-best --save-all --log-to-file --verbose 2
python train.py tcn stead1k5_6 regression --epochs 250 --lr 0.001 --lr-reducer --early-stopping --name tcn-stead1k5_6 --cudnn-lstm --save-best --save-all --log-to-file --verbose 2
python train.py tcn stead9k_6 regression --epochs 250 --lr 0.001 --lr-reducer --early-stopping --name tcn-stead9k_6 --cudnn-lstm --save-best --save-all --log-to-file --verbose 2

python train.py tcn speech8khz1k5_6 regression --epochs 250 --lr 0.001 --lr-reducer --early-stopping --name tcn-speech8khz1k5_6 --use-max-stream --cudnn-lstm --save-best --save-all --log-to-file --verbose 2
python train.py tcn speech8khz9k_6 regression --epochs 250 --lr 0.001 --lr-reducer --early-stopping --name tcn-speech8khz9k_6 --use-max-stream --cudnn-lstm --save-best --save-all --log-to-file --verbose 2
python train.py tcn emg1k5_6 regression --epochs 250 --lr 0.001 --lr-reducer --early-stopping --name tcn-emg1k5_6 --use-max-stream --cudnn-lstm --save-best --save-all --log-to-file --verbose 2
python train.py tcn emg9k_6 regression --epochs 250 --lr 0.001 --lr-reducer --early-stopping --name tcn-emg9k_6 --use-max-stream --cudnn-lstm --save-best --save-all --log-to-file --verbose 2
python train.py tcn sp5001k5_6 regression --epochs 250 --lr 0.001 --lr-reducer --early-stopping --name tcn-sp5001k5_6 --use-max-stream --cudnn-lstm --save-best --save-all --log-to-file --verbose 2
python train.py tcn sp5009k_6 regression --epochs 250 --lr 0.001 --lr-reducer --early-stopping --name tcn-sp5009k_6 --use-max-stream --cudnn-lstm --save-best --save-all --log-to-file --verbose 2