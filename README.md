## Intra-domain and cross-domain transfer learning for time series data – How transferable are the features?

This repository contains Python scripts necessary to reproduce the results from the paper "Intra-domain and cross-domain transfer learning for time series data – How transferable are the features?" It also contains the data and the results we obtained.

**Steps to reproduce the results:**

1. Start by cloning this repository to your local computer. Once cloned, open the terminal and navigate to the root directory of the cloned repository.

2. Recreate virtual Anaconda environment from the *environment.yml* file:

> conda env create -f environment.yml

Activate newly created Anaconda environment:

> conda activate tlenv

Note: this environment was exported on Linux-based OS. If you try to import it on Windows or macOS, the import will fail because some packages have different build numbers on different operating system. In this case you must manually install required packages from the *environemt.yml* file.

3. Download required datasets to your local computer.

   3.1. Lomax dataset
   
   Download Version 1 from <https://zenodo.org/record/5040865>
   
   This dataset comes as a single HDF5 file.
   
   3.2. LEN-DB dataset
   
   Download Version 1 published on February, 6th 2020. from <https://zenodo.org/record/3648232>
   
   This dataset comes as a single HDF5 file.

   3.3. STEAD dataset
   
   Download STEAD dataset from the <https://github.com/smousavi05/STEAD>
   
   For our experiment we downloaded this dataset on 15.11.2019. and the file was named waveforms_11_13_19.hdf5

   3.4. Speech commands dataset v0.0.2
   
   Download it from <http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz> and unpack to an empty directory.

   3.5. EMG dataset
   
   Download Pinch.zip from <https://zenodo.org/record/3194792> and extract it.

   3.6. S&P 500 dataset
   
   Download historical prices for S&P 500 (^GSPC) stock from Yahoo finance (<https://finance.yahoo.com/quote/^GSPC/>)
   
   Set 30th December 1927. as a start date, and 27th November 2018 as end date.

4. Navigate into the *scripts* directory:

> cd scripts

5.    Filter Lomax test set to contain only the instances that were correctly classified by the convolutional neural network (CNN) model used in the original study (87% accuracy) by executing the *filter_lomax.py*:

   > python filter_lomax.py &lt;path to the downloaded HDF5 file> &lt;path to save filtered HDF5 file>
   
   > Example: python filter_lomax.py /home/user/Downloads/ConvNetQuake_INGV_dataset.hdf5 /home/user/lomax_filtered.hdf5

6. Preprocess downloaded datasets to generate train, validation and test sets in NumPy format.

> python preprocess_lomax.py &lt;path to the Lomax filtered dataset from previous step> lomax

> python preprocess_lendb.py &lt;path to LEN-DB HDF5 file> lendb

> python preprocess_stead.py &lt;path to STEAD HDF5 file> stead 

> python preprocess_speech_commands_8khz.py &lt;path to the extracted directory from Speech commands archive> speech8khz

> python preprocess_emg.py &lt;path to the downloaded Pinch directory> emg

> python preprocess_sp500.py &lt;path to the single CSV file downloaded from Yahoo finance for S&P 500> sp500

These commands produce output inside *data* directory.

7. Generate small variants of given datasets with 1500 and 9000 training samples called 1k5 and 9k, respectively. In this process we only reduce the training set size, while validation and test sets are kept intact. These datasets are used to "simulate" real world scenario when only small quantities of data are available. Since we rerun entire experiment seven times, it was necessary to create seven small datasets for 1k5 and 9k variant for each downloaded dataset. In some cases we also duplicate channels for the reasons described in the paper.

Commands to perform this can be found in *create_small_datasets.sh*
These commands produce small datasets which are also located in *data* directory. To speed up the process, you can run these commands in parallel.

8. Create pre-trained (baseline) models that will later be used for transfer learning. This operation will produce 24 models (4 architectures x 6 datasets) in *models* directory.

To do that, execute commands from the *train_baseline_models.sh* script. If you have multiple GPUs, you can select which one you want to use by appending *--gpu n* option to the commands.

In our experiment, we created baseline models ten times and used the best ones for the reset of the experiment. If you wish to do the same, modify the script to perform the training desired number of times and then manually delete unnecessary models and preserve only the best models for the rest of the experiment.

9. Make predictions on test sets with baseline models to evaluate their performance. Execute *make_baseline_predictions.py* script, and once it is finished, execute *report_baseline_models.py*

This will produce a HTML report named *baseline_models.html* in *reports* directory. Now you can manually examine their performance and check how well they learned the tasks. We want to make sure that pre-training is successful, but it is not crucial for the models to achieve state-of-the-art performance.

10. Create referent models by training models "from scratch" on small datasets. These models will be later compared to the models trained with transfer learning.

To do that, execute commands from the *train_referent_models.sh*

All commands are provided in a single file. However, executing them sequentially would take a long time so we encourage you to distribute them across multiple computers (commands do not depend on one another). If you have multiple GPUs, you can select which one you want to use by appending *--gpu n* option to the commands.

Each training is done seven times which gives at the end a total of 336 models (7 reruns x 4 architectures x 12 small datasets)

11. Create transfer learning models by fine-tuning pre-trained models on small datasets. 

To do that, execute all scripts from *tl_models_creation_scripts* directory.

For the sake of simplicity, necessary commands are divided into 1680 scripts, and each script contains 10 commands. Executing them sequentially would take a long time so we encourage you to distribute them across multiple computers (commands do not depend on one another). If you have multiple GPUs, you can select which one you want to use by appending *--gpu n* option to the commands.

Success of transfer learning depends on the chosen LR multiplier value so we test ten different values: {0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0}.

Models that were pre-trained on one dataset are fine-tuned on other datasets and never on the one that was used for pre-training. This step was also rerun seven times.

This steps produces 16800 (10 LR multipliers x 7 reruns x 24 baseline models x 5 possible target datasets x 2 variants of target dataset) TL models fine-tuned on small datasets.

12. Make predictions on test sets with referent models. Execute the following commands to do that. If you wish, you can also distribute these across seven computers. Each command generates predictions for specified rerun of the experiment. If necessary, you can specify *--gpu n* option to use specific GPU.

> python make_multiple_referent_predictions.py 0

> python make_multiple_referent_predictions.py 1

> python make_multiple_referent_predictions.py 2

> python make_multiple_referent_predictions.py 3

> python make_multiple_referent_predictions.py 4

> python make_multiple_referent_predictions.py 5

> python make_multiple_referent_predictions.py 6

13. Make predictions on test sets with transfer learning models. 

To do that, execute commands from the *make_tl_predictions.sh* file. It is advisable to distribute these commands across multiple computers to speed up the process.

14. Compute metrics/scores for referent models. This is done by running the following command:

> python compute_multiple_runs_referent_metrics.py

15. Compute metrics/scores for TL models. This is done by running the following commands:

> python compute_multiple_runs_tl_metrics.py 0.01

> python compute_multiple_runs_tl_metrics.py 0.05

> python compute_multiple_runs_tl_metrics.py 0.1

> python compute_multiple_runs_tl_metrics.py 0.25

> python compute_multiple_runs_tl_metrics.py 0.5

> python compute_multiple_runs_tl_metrics.py 0.75

> python compute_multiple_runs_tl_metrics.py 1.0

> python compute_multiple_runs_tl_metrics.py 1.25

> python compute_multiple_runs_tl_metrics.py 1.5

> python compute_multiple_runs_tl_metrics.py 2.0

You can run these commands in parallel on the same machine. These commands do not use GPU.

16. Use the following commands to generate tables and figures presented in the paper:

> python report_multiple_runs_wins_loses_performance.py

> python report_multiple_runs_wins_loses_convergence_rate.py

> python report_multiple_runs_domains_performance.py

> python report_multiple_runs_domains_convergence_rate.py

> python report_multiple_runs_wins_loses_per_architecture.py

> python report_multiple_runs_performance_gain_vs_lr.py

> python report_multiple_runs_performance_per_model.py

> python report_multiple_runs_convergence_rate_per_model.py

Some of these scripts produce results in the *reports* directory, while others output results on the screen.

**Generating reports from our data**

The *our_models* directory contains files with calculated metrics for the models we trained (saved model weights were not uploaded because they would take too much space). You can use this data to generate reports without conducting your own training. To do this, simply copy the content of *our_models* directory into the *models* directory and run any script from the 16th step of previous section.

You can also write your own scripts to generate reports you are interested in.

**Our reports**

The reports we generated that were published in our paper can be found inside the *our_reports* directory. In all our reports, colored cells are found to be significant by the Wilcoxon signed-rank test (significance level of 0.05) and Benjamini-Krieger-Yekutieli correction is applied to control the false discovery rate (except in two cases). Green color represents that transfer learning models outperformed referent (non transfer learning) models, while red color indicates the opposite.

Note: scripts that count wins and loses do not produce any files as the output so their output is not contained in this directory.

Tables and diagrams included in the paper:
- domains_comparison_performance.tex - average difference in performance between transfer learning and referent models for each pair (source domain, target domain).
- domains_comparison_convergence_rate.tex - average difference in convergence rate between transfer learning and referent models for each pair (source domain, target domain).
- performance_gain_vs_lr_multipliers.png - impact of learning rate multiplier on the transfer learning predictive performance gain for each model.

Tables in the supplementary materials:
 - performance_per_model.tex - average percentual gain in predictive performance across seven reruns of the experiment for each triplet (model, source domain, target domain). Benjamini-Krieger-Yekutieli correction is not performed in this case.
 - convergence_rate_per_model.tex - average percentual gain in convergence rate across seven reruns of the experiment for each triplet (model, source domain, target domain). Benjamini-Krieger-Yekutieli correction is not performed in this case.

**Examining specific pairs of source and target domains**

If you want to perform transfer learning between two specific source and target domains, you can do that much easily than replicating entire experiment.

For example, if you want to pre-train MagNet model on LEN-DB dataset and fine-tune it on STEAD, you can achieve that with the following two commands:

1. pre-training
> python train.py magnet lendb regression --epochs 250 --lr 0.001 --lr-reducer --early-stopping --name magnet-lendb --use-max-stream --save-best --cudnn-lstm

2. transfer learning
> python train.py magnet stead1k5_0 regression --epochs 250 --lr 0.001 --lr-reducer --early-stopping --name magnet-lendb-stead1k5_0 --lr-multiplier 0.5 --weights magnet-lendb/ --cudnn-lstm --save-best --save-all --log-to-file --verbose 2

3. referent model training
> python train.py magnet stead1k5_0 regression --epochs 250 --lr 0.001 --lr-reducer --early-stopping --name magnet-stead1k5_0 --lr-multiplier 0.5 --cudnn-lstm --save-best --save-all --log-to-file --verbose 2

These models will be saved into *models* directory. To evaluate their performances, load both of them (*magnet-lendb-stead1k5_0* and *magnet-stead1k5_0*) and evaluate them on test set.


Used arguments:

> --epochs - maximal number of epochs for training

> --lr 0.001 - initial learning rate

> --lr-reducer - if specified, learning rate is decreased during training

> --early-stopping - stops training when validation loss does not decrease for 10 consecutive epochs

> --name - name under which to save trained model. Directory with this name will be created inside models directory

> --lr-multiplier - learning rate multiplier which is used during fine-tuning of convolutional layers

> --weights - defines whose weights to load for fine-tuning

> --cudnn-lstm - uses fast CuDNN implementation of LSTM layers. Omit this option if for some reason you do not wat to use CuDNN implementation.

> --save-best - save the best models during training

> --save-all - save model at each epoch

> --log-to-file - redirect standard output and standard error to the text file inside models directory

> --verbose - specifies how detailed output during training you want

Execute python *train.py -h* to check more details on these and other available options.

*Feel free to contact me at <erik.otovic@gmail.com>*
