# Cross-Modal Multitask Transformer for End-to-End Multimodal Aspect-Based Sentiment Analysis

#### Author: Li YANG,  yang0666@e.ntu.edu.sg

#### The Corresponding Paper: 
##### Cross-modal multitask transformer for end-to-end multimodal aspect-based sentiment analysis 
##### [[https://www.sciencedirect.com/science/article/abs/pii/S0306457324000840](https://www.sciencedirect.com/science/article/pii/S0306457322001479)](https://www.sciencedirect.com/science/article/abs/pii/S0306457322001479)

##### The framework of the CMMT model:  
![alt text]<img width="709" alt="Screenshot 2024-04-10 at 10 38 04 AM" src="https://github.com/yangli-hub/CMMT-Code/assets/70850281/4cd2be80-f1a5-4dcb-a3b3-255c24cd184c">



## Data
- The preprocessed CoNLL format files are provided in this repo. For each tweet, the first line is its image id, and the following lines are its textual contents.
- Step 1：Download each tweet's associated images via this link (https://drive.google.com/file/d/1PpvvncnQkgDNeBMKVgG2zFYuRhbL873g/view), and then put the associated images into folders "./image_data/twitter2015/" and "./image_data/twitter2017/";
- The politician dataset can be get via: https://drive.google.com/file/d/1oa029MLk8I_J99pxBs7X9RaIbUHhhTNG/view?usp=sharing
- Step 2: Download the image label file via this link(https://drive.google.com/drive/folders/17_ifeBqnCpHkd0Ns8cNcT-Q-Q5OxtSmZ?usp=sharing), and then put the associaled image label files into folder "./ANP_data/"
- Step 3: Download the pre-trained ResNet-152 via this link (https://download.pytorch.org/models/resnet152-b121ed2d.pth), and put the pre-trained ResNet-152 model under the folder './model/resnet/" 

- Step 4: Download the pre-trained roberta-base-cased from huggingface and put the pre-trained roberta model under the folder "./model/roberta-base-cased/"
- Step 5: ANP information can be downloaded via https://drive.google.com/drive/folders/1UaeSYJQCQzszRmBWdhA11LqXnWousn4G?usp=share_link

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

## Citation Information:
Yang, L., Na, J. C., & Yu, J. (2022). Cross-modal multitask transformer for end-to-end multimodal aspect-based sentiment analysis. Information Processing & Management, 59(5), 103038.
