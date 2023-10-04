# Reward Imputation with Sketching for Contextual Batched Bandits

This repository is the official implementation of ***Reward Imputation with Sketching for Contextual Batched Bandits***.

This paper was accepted by NeurIPS 2023.

## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```

## Training

To train models except DFM-S in the paper, run this command:

```train
python algo_main.py  Algorithm_name
```
To train DFM-S in the paper, run this command:

```train
python algo_main2.py
```
We recommend you tuning hyper-parameters by using [nni module](https://github.com/microsoft/nni). In our experiments, we use nni to tune hyper-parameters.


## Evaluation

Because we calculate average reward in each episode, you can export reward data using nni after running code.
To export reward data, run:

```eval
nnictl experiment export [experiment_id] --filename [file_path] --type json --intermediate
```

# Connect

If you have any questions, please contact us at the email address zhangx89@ruc.edu.cn, or submit an issue here.