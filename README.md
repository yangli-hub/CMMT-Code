# Cross-Modal Multitask Transformer for End-to-End Multimodal Aspect-Based Sentiment Analysis

Author

Li YANG

yang0666@e.ntu.edu.sg

June 25, 2022

## Data
- The preprocessed CoNLL format files are provided in this repo. For each tweet, the first line is its image id, and the following lines are its textual contents.
- Step 1：Download each tweet's associated images via this link (https://drive.google.com/file/d/1PpvvncnQkgDNeBMKVgG2zFYuRhbL873g/view), and then put the associated images into folders "./image_data/twitter2015/" and "./image_data/twitter2017/";
- Step 2: Download the image label file via this link(https://drive.google.com/file/d/17v9shne9W1j0wwgDsCCTJ4zUsKFSUxEw/view?usp=sharing), and then put the associaled image label files into folder "./ANP_data/"
- Step 3: Download the pre-trained ResNet-152 via this link (https://download.pytorch.org/models/resnet152-b121ed2d.pth), and put the pre-trained ResNet-152 model under the folder './model/resnet/" 

- Step 4: Download the pre-trained roberta-base-cased and put the pre-trained roberta model under the folder "./model/roberta-base-cased/"  


## Requirement
* PyTorch 1.0.0
* Python 3.7 
* pytorch-crf 0.7.2

## Code Usage

### Training for CMMT
- This is the training code of tuning parameters on the dev set, and testing on the test set. Note that you can change "CUDA_VISIBLE_DEVICES=2" based on your available GPUs.

```sh
sh run_cmmt_crf.sh
```

- We show our running logs on twitter-2015, twitter-2017 and political twitter in the folder "log files". Note that the results are a little bit lower than the results reported in our paper, since the experiments were run on different servers.


## Acknowledgements
- Using these two datasets means you have read and accepted the copyrights set by Twitter and dataset providers.
- Most of the codes are based on the codes provided by huggingface: https://github.com/huggingface/transformers.
