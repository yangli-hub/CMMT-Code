from __future__ import absolute_import, division, print_function

import argparse
import csv
import logging
import os
import random
import json
import sys

import numpy as np
import torch
import torch.nn.functional as F
from my_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from my_bert.mner_modeling import (CONFIG_NAME, WEIGHTS_NAME,
                                              BertConfig, MTCCMBertForMMTokenClassificationCRF)
from my_bert.optimization import BertAdam, warmup_linear
from my_bert.tokenization import BertTokenizer
from seqeval.metrics import classification_report
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

import resnet.resnet as resnet
#from resnet.resnet import resnet
from resnet.resnet_utils import myResnet

from torchvision import transforms
from PIL import Image

from sklearn.metrics import precision_recall_fscore_support

from ner_evaluate import evaluate_each_class
from ner_evaluate import evaluate
from transformers import RobertaTokenizer, RobertaModel

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)


def image_process(image_path, transform):
    image = Image.open(image_path).convert('RGB')
    image = transform(image)
    return image


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


class MMInputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b, img_id, label=None, auxlabel=None,imagelabel= None): # yl add
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.img_id = img_id
        self.label = label
        self.auxlabel = auxlabel
        self.imagelabel = imagelabel


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id


class MMInputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, added_input_mask, segment_ids, img_feat, label_id, auxlabel_id, imagelabel):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.added_input_mask = added_input_mask
        self.segment_ids = segment_ids
        self.img_feat = img_feat
        self.label_id = label_id
        self.auxlabel_id = auxlabel_id
        self.imagelabel = imagelabel


def readfile(filename):
    '''
    read file
    return format :
    [ ['EU', 'B-ORG'], ['rejects', 'O'], ['German', 'B-MISC'], ['call', 'O'], ['to', 'O'], ['boycott', 'O'], ['British', 'B-MISC'], ['lamb', 'O'], ['.', 'O'] ]
    '''
    f = open(filename)
    data = []
    sentence = []
    label= []
    for line in f:
        if len(line)==0 or line.startswith('-DOCSTART') or line[0]=="\n":
            if len(sentence) > 0:
                data.append((sentence,label))
                sentence = []
                label = []
            continue
        splits = line.split(' ')
        sentence.append(splits[0])
        label.append(splits[-1][:-1])

    if len(sentence) >0:
        data.append((sentence,label))
        sentence = []
        label = []

    print("The number of samples: "+ str(len(data)))
    return data


def mmreadfile(filename, image_filename, path_img):
    '''
    read file
    return format :
    [ ['EU', 'B-ORG'], ['rejects', 'O'], ['German', 'B-MISC'], ['call', 'O'], ['to', 'O'], ['boycott', 'O'], ['British', 'B-MISC'], ['lamb', 'O'], ['.', 'O'] ]
    '''
    transform = transforms.Compose([
        transforms.RandomCrop(224),  # args.crop_size, by default it is set to be 224
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225))])    
    with open(image_filename, 'r') as f:
        image_data = json.load(f)
    f = open(filename, encoding='utf-8')
    data = []
    imgs = []
    auxlabels = []
    sentence = []
    label= []
    auxlabel = []
    imagelabels = []
    imgid = ''
    count = 0
    # print(image_data.keys())
    for line in f:
        if line.startswith('IMGID:'):
            imgid = line.strip().split('IMGID:')[1]+'.jpg'
            continue
        if line[0]=="\n":
            if len(sentence) > 0:
                data.append((sentence,label))
                imgs.append(imgid)
                image_path = os.path.join(path_img, imgid)
                if not os.path.exists(image_path):
                    print(image_path)
                try:
                    image = image_process(image_path, transform)
                except:
                    # print('image has problem!')
                    imgid = '17_06_4705.jpg'                                        

                image_label = image_data.get(imgid)
                if image_label == None:
                    count += 1
                    #print(sentence)
                    #print(label)
                    #print(imgid)
                auxlabels.append(auxlabel)
                imagelabels.append(image_label)
                sentence = []
                label = []
                imgid = ''
                auxlabel = []
            continue
        splits = line.split('\t')
        sentence.append(splits[0])
        cur_label = splits[1] #splits[-1][:-1]   # yl add
        if cur_label == 'B-OTHER':
            cur_label = 'B-MISC'
        elif cur_label == 'I-OTHER':
            cur_label = 'I-MISC'
        label.append(cur_label)
        auxlabel.append(cur_label)
        #auxlabel.append(splits[2][:-1]) # yl add
    
    print("The number of samples with NULL image labels: "+ str(count))
    if len(sentence) >0:
        data.append((sentence,label))
        imgs.append(imgid)
        auxlabels.append(auxlabel)
        imagelabels.append(image_label)
        sentence = []
        label = []
        auxlabel = []

    print("The number of samples: "+ str(len(data)))
    print("The number of images: "+ str(len(imgs)))
    return data, imgs, auxlabels, imagelabels


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir,image_filename,path_img): #yl add
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir,image_filename,path_img): #yl add
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        return readfile(input_file)

    def _read_mmtsv(cls, input_file, image_filename, path_img, quotechar=None):
        """Reads a tab separated value file."""
        return mmreadfile(input_file, image_filename, path_img) # yl add 


class MNERProcessor(DataProcessor):
    """Processor for the CoNLL-2003 data set."""

    def get_train_examples(self, data_dir, image_filename, path_img): # yl add 
        """See base class."""
        data, imgs, auxlabels,imagelabels = self._read_mmtsv(os.path.join(data_dir, "train.txt"), image_filename, path_img) # yl add 
        return self._create_examples(data, imgs, auxlabels, imagelabels, "train")

    def get_dev_examples(self, data_dir,image_filename,path_img): # yl add 
        """See base class."""
        data, imgs, auxlabels,imagelabels = self._read_mmtsv(os.path.join(data_dir, "valid.txt"), image_filename, path_img) # yl add 
        return self._create_examples(data, imgs, auxlabels, imagelabels, "dev")

    def get_test_examples(self, data_dir,image_filename,path_img): # yl add 
        """See base class."""
        data, imgs, auxlabels, imagelabels = self._read_mmtsv(os.path.join(data_dir, "test.txt"), image_filename, path_img) # yl add 
        return self._create_examples(data, imgs, auxlabels, imagelabels, "test")


    def get_labels(self):
        return ["O", "B-NEU", "I-NEU", "B-POS", "I-POS", "B-NEG", "I-NEG","X","<s>","</s>"]

    ### modify
    def get_auxlabels(self):
        return ["O", "B-NEU", "I-NEU", "B-POS", "I-POS", "B-NEG", "I-NEG", "X", "<s>", "</s>"] #yl add  ["O", "B-AE", "I-AE", "B-OE", "I-OE", "X", "<s>", "</s>"] 
    #def get_auxlabels(self):
        #return ["O", "B", "I", "X", "[CLS]", "[SEP]"]

    ### modify
    def get_start_label_id(self):
        label_list = self.get_labels()
        label_map = {label: i for i, label in enumerate(label_list, 1)}
        return label_map['<s>']

    def get_stop_label_id(self):
        label_list = self.get_labels()
        label_map = {label: i for i, label in enumerate(label_list, 1)}
        return label_map['</s>']

    def _create_examples(self, lines, imgs, auxlabels, imagelabels, set_type): # yl add
        examples = []
        for i, (sentence, label) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = ' '.join(sentence)
            text_b = None
            img_id = imgs[i]
            label = label
            auxlabel = auxlabels[i]
            imagelabel = imagelabels[i]
            examples.append(MMInputExample(guid=guid, text_a=text_a, text_b=text_b, img_id=img_id, label=label, auxlabel=auxlabel, imagelabel = imagelabel))
        return examples


def convert_examples_to_features(examples, label_list, max_seq_length, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""

    label_map = {label : i for i, label in enumerate(label_list,1)}
    
    features = []
    for (ex_index,example) in enumerate(examples):
        textlist = example.text_a.split(' ')
        labellist = example.label
        tokens = []
        labels = []
        for i, word in enumerate(textlist):
            token = tokenizer.tokenize(word)
            tokens.extend(token)
            label_1 = labellist[i]
            for m in range(len(token)):
                if m == 0:
                    labels.append(label_1)
                else:
                    labels.append("X")
        if len(tokens) >= max_seq_length - 1:
            tokens = tokens[0:(max_seq_length - 2)]
            labels = labels[0:(max_seq_length - 2)]
        ntokens = []
        segment_ids = []
        label_ids = []
        ntokens.append("<s>")
        segment_ids.append(0)
        label_ids.append(label_map["<s>"])
        for i, token in enumerate(tokens):
            ntokens.append(token)
            segment_ids.append(0)
            label_ids.append(label_map[labels[i]])
        ntokens.append("</s>")
        segment_ids.append(0)
        label_ids.append(label_map["</s>"])
        input_ids = tokenizer.convert_tokens_to_ids(ntokens)
        input_mask = [1] * len(input_ids)
        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)
            label_ids.append(0)
        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert len(label_ids) == max_seq_length
        
        if ex_index < 2:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("tokens: %s" % " ".join(
                    [str(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logger.info(
                    "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            logger.info("label: %s" % " ".join([str(x) for x in label_ids]))

        features.append(
                InputFeatures(input_ids=input_ids,
                              input_mask=input_mask,
                              segment_ids=segment_ids,
                              label_id=label_ids))
    return features


def convert_mm_examples_to_features(examples, label_list, auxlabel_list, max_seq_length, tokenizer, crop_size, path_img):
    """Loads a data file into a list of `InputBatch`s."""

    label_map = {label: i for i, label in enumerate(label_list, 1)}
    auxlabel_map = {label: i for i, label in enumerate(auxlabel_list, 1)}

    features = []
    count = 0

    transform = transforms.Compose([
        transforms.RandomCrop(crop_size),  # args.crop_size, by default it is set to be 224
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225))])

    for (ex_index, example) in enumerate(examples):
        textlist = example.text_a.split(' ')
        labellist = example.label
        auxlabellist = example.auxlabel
        imagelabellist = example.imagelabel
        imagelabellist = dict(sorted(imagelabellist.items()))
        imagelabel_value =[0]* len(imagelabellist)           
        for i, (k, v) in enumerate(imagelabellist.items()):
            imagelabel_value[i]= v
        tokens = []
        labels = []
        auxlabels = []
        for i, word in enumerate(textlist):
            word = " "+ word
            token = tokenizer.tokenize(word)
            tokens.extend(token)
            label_1 = labellist[i]
            auxlabel_1 = auxlabellist[i]
            for m in range(len(token)):
                if m == 0:
                    labels.append(label_1)
                    auxlabels.append(auxlabel_1)
                else:
                    labels.append("X")
                    auxlabels.append("X")
        if len(tokens) >= max_seq_length - 1:
            tokens = tokens[0:(max_seq_length - 2)]
            labels = labels[0:(max_seq_length - 2)]
            auxlabels = auxlabels[0:(max_seq_length - 2)]
        ntokens = []
        segment_ids = []
        label_ids = []
        auxlabel_ids = []
        ntokens.append("<s>")  
        segment_ids.append(0)
        label_ids.append(label_map["<s>"]) 
        auxlabel_ids.append(auxlabel_map["<s>"])
        for i, token in enumerate(tokens):
            ntokens.append(token)
            segment_ids.append(0)
            label_ids.append(label_map[labels[i]])
            auxlabel_ids.append(auxlabel_map[auxlabels[i]])
        ntokens.append("</s>") 
        segment_ids.append(0)
        label_ids.append(label_map["</s>"])  
        auxlabel_ids.append(auxlabel_map["</s>"]) 
        input_ids = tokenizer.convert_tokens_to_ids(ntokens)
        input_mask = [1] * len(input_ids)
        added_input_mask = [1] * (len(input_ids) + 49)  # 1 or 49 is for encoding regional image representations

        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            added_input_mask.append(0)
            segment_ids.append(0)
            label_ids.append(0)
            auxlabel_ids.append(0)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert len(label_ids) == max_seq_length
        assert len(auxlabel_ids) == max_seq_length

        image_name = example.img_id
        image_path = os.path.join(path_img, image_name)

        if not os.path.exists(image_path):
            print(image_path)
        try:
            image = image_process(image_path, transform)
        except:
            count += 1
            # print('image has problem!')
            image_path_fail = os.path.join(path_img, '17_06_4705.jpg')
            image = image_process(image_path_fail, transform)

        if ex_index < 2:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("tokens: %s" % " ".join(
                [str(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logger.info(
                "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            logger.info("label: %s" % " ".join([str(x) for x in label_ids]))
            logger.info("auxlabel: %s" % " ".join([str(x) for x in auxlabel_ids]))

        features.append(
            MMInputFeatures(input_ids=input_ids, input_mask=input_mask, added_input_mask=added_input_mask,
                          segment_ids=segment_ids, img_feat=image, label_id=label_ids, auxlabel_id=auxlabel_ids, imagelabel= imagelabel_value))

    print('the number of problematic samples: ' + str(count))

    return features


def macro_f1(y_true, y_pred):
    p_macro, r_macro, f_macro, support_macro \
      = precision_recall_fscore_support(y_true, y_pred, average='macro')
    f_macro = 2*p_macro*r_macro/(p_macro+r_macro)
    return p_macro, r_macro, f_macro

import datetime
def main():
    start_time = datetime.datetime.now().strftime('%m-%d-%Y-%H-%M-%S_')
    parser = argparse.ArgumentParser()

    parser.add_argument('--cuda_id', type=str, default='0',
                        help='Choose which GPUs to run')                   
    parser.add_argument("--bert_model", default="./model/roberta-base-cased", type=str,
                        help="pre-trained model selected in the list: roberta-base-cased, "
                        "bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased, "
                        "bert-base-multilingual-cased, bert-base-chinese.")

    parser.add_argument("--task_name",
                        default= "twitter2015",  #twitter2017
                        type=str,
                        required=True,
                        help="The name of the task to train.")
    parser.add_argument("--cache_dir",
                        default="",
                        type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")
    parser.add_argument("--max_seq_length",
                        default=128,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--do_train",
                        action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval",
                        action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_lower_case",
                        action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--train_batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size",
                        default=16,
                        type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--learning_rate",
                        default=3e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs",
                        default=25.0,
                        type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--seed',
                        type=int,
                        default=64,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--fp16',
                        action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--loss_scale',
                        type=float, default=0,
                        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                             "0 (default value): dynamic loss scaling.\n"
                             "Positive power of 2: static loss scaling value.\n")
    parser.add_argument('use_roberta', default=True, action = 'store_true')

    parser.add_argument('--mm_model', default='MTCCMBert', help='model name')
    parser.add_argument('--layer_num1', type=int, default=1, help='number of txt2img layer')
    parser.add_argument('--layer_num2', type=int, default=1, help='number of img2txt layer')
    parser.add_argument('--layer_num3', type=int, default=1, help='number of txt2txt layer')
    parser.add_argument('--fine_tune_cnn', action='store_true', help='fine tune pre-trained CNN if True')
    parser.add_argument('--resnet_root', default='./model/resnet', help='path the pre-trained cnn models')
    parser.add_argument('--crop_size', type=int, default=224, help='crop size of image')
    parser.add_argument('--path_image', default='./pytorch-pretrained-BERT/twitter_subimages/', help='path to images')
    parser.add_argument('--server_ip', type=str, default='', help="Can be used for distant debugging.")
    parser.add_argument('--server_port', type=str, default='', help="Can be used for distant debugging.")
    parser.add_argument('--alpha', type=float, default=1)
    parser.add_argument('--beta', type=float, default=1)
    parser.add_argument('--dropout_rate', type=float, default=0.2)
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] =args.cuda_id
    

    if args.task_name == "twitter2017": # this refers to twitter-2017 dataset
        args.path_image = "/mnt/nfs-storage-titan/alienware/intern18_snap/multi_modal_ABSA_pytorch_naacl/multi_modal_ABSA_pytorch_bilinear/twitter_subimages/"
        args.data_dir = "./data/twitter2017"
        args.image_filename = "./ANP_data/image_output2017.json"
        args.output_dir = start_time + "_twitter2017_output/"
    elif args.task_name == "twitter2015":  # this refers to twitter-2015 dataset
        args.path_image = "/mnt/nfs-storage-titan/alienware/intern18_snap/multi_modal_ABSA_pytorch_naacl/multi_modal_ABSA_pytorch_bilinear/twitter15_images/"
        args.data_dir = "./data/twitter2015"
        args.image_filename = "./ANP_data/image_output2015.json"
        args.output_dir = start_time +  "_twitter2015_output/"
    
    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd
        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()
    
    processors = {
        "twitter2015": MNERProcessor,
        "twitter2017": MNERProcessor
        }

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')
    logger.info("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
        device, n_gpu, bool(args.local_rank != -1), args.fp16))

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
                            args.gradient_accumulation_steps))
    
    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    args.do_train = True
    args.do_eval = True
    if not args.do_train and not args.do_eval:
        raise ValueError("At least one of `do_train` or `do_eval` must be True.")

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train:
        raise ValueError("Output directory ({}) already exists and is not empty.".format(args.output_dir))
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    task_name = args.task_name.lower()

    if task_name not in processors:
        raise ValueError("Task not found: %s" % (task_name))

    processor = processors[task_name]()
    label_list = processor.get_labels()
    auxlabel_list = processor.get_auxlabels()
    num_labels = len(label_list)+1 # label 0 corresponds to padding, label in label_list starts from 1
    auxnum_labels = len(auxlabel_list)+1 # label 0 corresponds to padding, label in label_list starts from 1

    start_label_id = processor.get_start_label_id()
    stop_label_id = processor.get_stop_label_id() 

    trans_matrix = np.zeros((auxnum_labels,num_labels), dtype=float)
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

    train_examples = None
    num_train_optimization_steps = None
    if args.do_train:
        train_examples = processor.get_train_examples(args.data_dir, args.image_filename, args.path_image) # yl add
        num_train_optimization_steps = int(
            len(train_examples) / args.train_batch_size / args.gradient_accumulation_steps) * args.num_train_epochs
        if args.local_rank != -1:
            num_train_optimization_steps = num_train_optimization_steps // torch.distributed.get_world_size()
    
    # Prepare model
    cache_dir = args.cache_dir if args.cache_dir else os.path.join(str(PYTORCH_PRETRAINED_BERT_CACHE), 'distributed_{}'.format(args.local_rank))

    if args.mm_model == 'MTCCMBert':
        model = MTCCMBertForMMTokenClassificationCRF.from_pretrained(args.bert_model,args.use_roberta,
              cache_dir=cache_dir, layer_num1=args.layer_num1, layer_num2=args.layer_num2, layer_num3=args.layer_num3,
              num_labels = num_labels, auxnum_labels = auxnum_labels,dropout_rate=args.dropout_rate)
    if args.use_roberta:
        roberta_dict = torch.load('./model/roberta-base-cased/pytorch_model.bin')  #changed to large or base
        new_state_dict = model.state_dict()
        miss_keys = []
        for k in new_state_dict.keys():
            if k in roberta_dict.keys():
                new_state_dict[k] = roberta_dict[k]
            else:
                miss_keys.append(k)
        if len(miss_keys) > 0:
            logger.info('miss keys: {}'.format(miss_keys))
        model.load_state_dict(new_state_dict)

    else:
        print('please define your MNER Model')

    net = getattr(resnet, 'resnet152')()
    net.load_state_dict(torch.load(os.path.join(args.resnet_root, 'resnet152.pth')))
    encoder = myResnet(net, args.fine_tune_cnn, device)
    if args.fp16:
        model.half()
        encoder.half()
    model.to(device)
    encoder.to(device)
    if args.local_rank != -1:
        try:
            from apex.parallel import DistributedDataParallel as DDP
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

        model = DDP(model)
        encoder = DDP(encoder)
    elif n_gpu > 1:
        model = torch.nn.DataParallel(model)
        encoder = torch.nn.DataParallel(encoder)

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
    if args.fp16:
        try:
            from apex.optimizers import FP16_Optimizer
            from apex.optimizers import FusedAdam
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

        optimizer = FusedAdam(optimizer_grouped_parameters,
                              lr=args.learning_rate,
                              bias_correction=False,
                              max_grad_norm=1.0)
        if args.loss_scale == 0:
            optimizer = FP16_Optimizer(optimizer, dynamic_loss_scale=True)
        else:
            optimizer = FP16_Optimizer(optimizer, static_loss_scale=args.loss_scale)

    else:
        optimizer = BertAdam(optimizer_grouped_parameters,
                             lr=args.learning_rate,
                             warmup=args.warmup_proportion,
                             t_total=num_train_optimization_steps)

    global_step = 0
    nb_tr_steps = 0
    tr_loss = 0

    output_model_file = os.path.join(args.output_dir, WEIGHTS_NAME)
    output_config_file = os.path.join(args.output_dir, CONFIG_NAME)
    output_encoder_file = os.path.join(args.output_dir, "pytorch_encoder.bin")

    if args.do_train:
        train_features = convert_mm_examples_to_features(
            train_examples, label_list, auxlabel_list, args.max_seq_length, tokenizer, args.crop_size, args.path_image)
        all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
        all_added_input_mask = torch.tensor([f.added_input_mask for f in train_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
        all_img_feats = torch.stack([f.img_feat for f in train_features])
        all_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.long)
        all_auxlabel_ids = torch.tensor([f.auxlabel_id for f in train_features], dtype=torch.long)
        all_imagelabel = torch.tensor([f.imagelabel for f in train_features], dtype=torch.float)

        train_data = TensorDataset(all_input_ids, all_input_mask, all_added_input_mask, all_segment_ids, all_img_feats,
                                  all_label_ids, all_auxlabel_ids,all_imagelabel)

        if args.local_rank == -1:
            train_sampler = RandomSampler(train_data)
        else:
            train_sampler = DistributedSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)

        eval_examples = processor.get_dev_examples(args.data_dir, args.image_filename, args.path_image) # yl add
        eval_features = convert_mm_examples_to_features(
            eval_examples, label_list, auxlabel_list, args.max_seq_length, tokenizer, args.crop_size, args.path_image)

        all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
        all_added_input_mask = torch.tensor([f.added_input_mask for f in eval_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
        all_img_feats = torch.stack([f.img_feat for f in eval_features])
        all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)
        all_auxlabel_ids = torch.tensor([f.auxlabel_id for f in eval_features], dtype=torch.long)
        all_imagelabel = torch.tensor([f.imagelabel for f in eval_features], dtype=torch.float)
        eval_data = TensorDataset(all_input_ids, all_input_mask, all_added_input_mask, all_segment_ids, all_img_feats,
                                  all_label_ids, all_auxlabel_ids,all_imagelabel)

        # Run prediction for full data
        eval_sampler = SequentialSampler(eval_data)
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

        test_eval_examples = processor.get_test_examples(args.data_dir, args.image_filename, args.path_image) # yl add 
        test_eval_features = convert_mm_examples_to_features(
            test_eval_examples, label_list, auxlabel_list, args.max_seq_length, tokenizer, args.crop_size, args.path_image)
        all_input_ids = torch.tensor([f.input_ids for f in test_eval_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in test_eval_features], dtype=torch.long)
        all_added_input_mask = torch.tensor([f.added_input_mask for f in test_eval_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in test_eval_features], dtype=torch.long)
        all_img_feats = torch.stack([f.img_feat for f in test_eval_features])
        all_label_ids = torch.tensor([f.label_id for f in test_eval_features], dtype=torch.long)
        all_auxlabel_ids = torch.tensor([f.auxlabel_id for f in test_eval_features], dtype=torch.long)
        all_imagelabel = torch.tensor([f.imagelabel for f in test_eval_features], dtype=torch.float)

        test_eval_data = TensorDataset(all_input_ids, all_input_mask, all_added_input_mask, all_segment_ids, all_img_feats,
                                  all_label_ids, all_auxlabel_ids,all_imagelabel)
        # Run prediction for full data
        test_eval_sampler = SequentialSampler(test_eval_data)
        test_eval_dataloader = DataLoader(test_eval_data, sampler=test_eval_sampler, batch_size=args.eval_batch_size)

        max_dev_f1 = 0.0
        max_test_f1 = 0.0
        best_dev_epoch = 0
        best_test_epoch = 0
        logger.info("***** Running training *****")
        for train_idx in trange(int(args.num_train_epochs), desc="Epoch"):
            logger.info("********** Epoch: " + str(train_idx) + " **********")
            logger.info("  Num examples = %d", len(train_examples))
            logger.info("  Batch size = %d", args.train_batch_size)
            logger.info("  Num steps = %d", num_train_optimization_steps)
            model.train()
            encoder.train()
            encoder.zero_grad()
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0
            for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
                batch = tuple(t.to(device) for t in batch)
                input_ids, input_mask, added_input_mask, segment_ids, img_feats, label_ids, auxlabel_ids, imagelabel = batch
                with torch.no_grad():
                    imgs_f, img_mean, img_att = encoder(img_feats)
                trans_matrix = torch.tensor(trans_matrix).to(device)
                neg_log_likelihood = model(input_ids, segment_ids, input_mask, added_input_mask,
                                                              img_att, trans_matrix, imagelabel,args.alpha, args.beta,label_ids, auxlabel_ids)
                if n_gpu > 1:
                    neg_log_likelihood = neg_log_likelihood.mean() # mean() to average on multi-gpu.
                if args.gradient_accumulation_steps > 1:
                    neg_log_likelihood = neg_log_likelihood / args.gradient_accumulation_steps

                if args.fp16:
                    optimizer.backward(neg_log_likelihood)
                else:
                    neg_log_likelihood.backward()

                tr_loss += neg_log_likelihood.item()
                nb_tr_examples += input_ids.size(0)
                nb_tr_steps += 1
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    if args.fp16:
                        # modify learning rate with special warm up BERT uses
                        # if args.fp16 is False, BertAdam is used that handles this automatically
                        lr_this_step = args.learning_rate * warmup_linear(global_step/num_train_optimization_steps, args.warmup_proportion)
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = lr_this_step
                    optimizer.step()
                    optimizer.zero_grad()
                    global_step += 1

            logger.info("***** Running evaluation on Dev Set*****")
            logger.info("  Num examples = %d", len(eval_examples))
            logger.info("  Batch size = %d", args.eval_batch_size)

            model.eval()
            encoder.eval()
            eval_loss, eval_accuracy = 0, 0
            nb_eval_steps, nb_eval_examples = 0, 0

            y_true = []
            y_pred = []
            y_true_idx = []
            y_pred_idx = []
            label_map = {i: label for i, label in enumerate(label_list, 1)}
            label_map[0] = "PAD"
            for input_ids, input_mask, added_input_mask, segment_ids, img_feats, label_ids, auxlabel_ids,imagelabel  in tqdm(
                    eval_dataloader,
                    desc="Evaluating"):
                input_ids = input_ids.to(device)
                input_mask = input_mask.to(device)
                added_input_mask = added_input_mask.to(device)
                segment_ids = segment_ids.to(device)
                img_feats = img_feats.to(device)
                label_ids = label_ids.to(device)
                auxlabel_ids = auxlabel_ids.to(device)

                with torch.no_grad():
                    imgs_f, img_mean, img_att = encoder(img_feats)
                    predicted_label_seq_ids = model(input_ids, segment_ids, input_mask, added_input_mask, img_att,
                                                    trans_matrix, imagelabel,args.alpha, args.beta)

                logits = predicted_label_seq_ids
                label_ids = label_ids.to('cpu').numpy()
                input_mask = input_mask.to('cpu').numpy()
                for i, mask in enumerate(input_mask):
                    temp_1 = []
                    temp_2 = []
                    tmp1_idx = []
                    tmp2_idx = []
                    for j, m in enumerate(mask):
                        if j == 0:
                            continue
                        if m:
                            if label_map[label_ids[i][j]] != "X" and label_map[
                                label_ids[i][j]] != "</s>": 
                                temp_1.append(label_map[label_ids[i][j]])
                                tmp1_idx.append(label_ids[i][j])
                                temp_2.append(label_map[logits[i][j]])
                                tmp2_idx.append(logits[i][j])
                        else:
                            break
                    y_true.append(temp_1)
                    y_pred.append(temp_2)
                    y_true_idx.append(tmp1_idx)
                    y_pred_idx.append(tmp2_idx)

            # report = classification_report(y_true, y_pred, digits=4)
            sentence_list = []
            dev_data, imgs, _ ,_ = processor._read_mmtsv(os.path.join(args.data_dir, "valid.txt"),args.image_filename, args.path_image)
            for i in range(len(y_pred)):
                sentence = dev_data[i][0]
                sentence_list.append(sentence)
            reverse_label_map = {label: i for i, label in enumerate(label_list, 1)}
            acc, f1, p, r = evaluate(y_pred_idx, y_true_idx, sentence_list, reverse_label_map)
            logger.info("***** Dev Eval results *****")
            print("Overall: ", p, r, f1)
            per_f1, per_p, per_r = evaluate_each_class(y_pred_idx, y_true_idx, sentence_list, reverse_label_map, 'POS')
            print("Positive: ", per_p, per_r, per_f1)
            loc_f1, loc_p, loc_r = evaluate_each_class(y_pred_idx, y_true_idx, sentence_list, reverse_label_map, 'NEU')
            print("Neutral: ", loc_p, loc_r, loc_f1)
            org_f1, org_p, org_r = evaluate_each_class(y_pred_idx, y_true_idx, sentence_list, reverse_label_map, 'NEG')
            print("Negative: ", org_p, org_r, org_f1)
            F_score_dev = f1

            logger.info("***** Running Test evaluation *****")
            logger.info("  Num examples = %d", len(test_eval_examples))
            logger.info("  Batch size = %d", args.eval_batch_size)
            y_true = []
            y_pred = []
            y_true_idx = []
            y_pred_idx = []
            label_map = {i: label for i, label in enumerate(label_list, 1)}
            label_map[0] = "PAD"
            for input_ids, input_mask, added_input_mask, segment_ids, img_feats, label_ids, auxlabel_ids,imagelabel in tqdm(test_eval_dataloader,
                                                                                                   desc="Evaluating"):
                input_ids = input_ids.to(device)
                input_mask = input_mask.to(device)
                added_input_mask = added_input_mask.to(device)
                segment_ids = segment_ids.to(device)
                img_feats = img_feats.to(device)
                label_ids = label_ids.to(device)
                auxlabel_ids = auxlabel_ids.to(device)

                with torch.no_grad():
                    imgs_f, img_mean, img_att = encoder(img_feats)
                    predicted_label_seq_ids = model(input_ids, segment_ids, input_mask, added_input_mask, img_att,trans_matrix,imagelabel,args.alpha, args.beta)

                logits = predicted_label_seq_ids
                label_ids = label_ids.to('cpu').numpy()
                input_mask = input_mask.to('cpu').numpy()
                for i, mask in enumerate(input_mask):
                    temp_1 = []
                    temp_2 = []
                    tmp1_idx = []
                    tmp2_idx = []
                    for j, m in enumerate(mask):
                        if j == 0:
                            continue
                        if m:
                            if label_map[label_ids[i][j]] != "X" and label_map[
                                label_ids[i][j]] != "</s>": 
                                temp_1.append(label_map[label_ids[i][j]])
                                tmp1_idx.append(label_ids[i][j])
                                temp_2.append(label_map[logits[i][j]])
                                tmp2_idx.append(logits[i][j])
                        else:
                            break
                    y_true.append(temp_1)
                    y_pred.append(temp_2)
                    y_true_idx.append(tmp1_idx)
                    y_pred_idx.append(tmp2_idx)

            #report = classification_report(y_true, y_pred, digits=4)
            sentence_list = []
            test_data, imgs, _,_ = processor._read_mmtsv(os.path.join(args.data_dir, "test.txt"),args.image_filename, args.path_image)
            for i in range(len(y_pred)):
                sentence = test_data[i][0]
                sentence_list.append(sentence)

            reverse_label_map = {label: i for i, label in enumerate(label_list, 1)}
            acc, f1, p, r = evaluate(y_pred_idx, y_true_idx, sentence_list, reverse_label_map)
            logger.info("***** Test Eval results *****")
            print("Overall: ", p, r, f1)
            per_f1, per_p, per_r = evaluate_each_class(y_pred_idx, y_true_idx, sentence_list, reverse_label_map, 'POS')
            print("Positive: ", per_p, per_r, per_f1)
            loc_f1, loc_p, loc_r = evaluate_each_class(y_pred_idx, y_true_idx, sentence_list, reverse_label_map, 'NEU')
            print("Neutral: ", loc_p, loc_r, loc_f1)
            org_f1, org_p, org_r = evaluate_each_class(y_pred_idx, y_true_idx, sentence_list, reverse_label_map, 'NEG')
            print("Negative: ", org_p, org_r, org_f1)
            F_score_test = f1

            if F_score_dev > max_dev_f1:
                # Save a trained model and the associated configuration
                model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
                encoder_to_save = encoder.module if hasattr(encoder,
                                                            'module') else encoder  # Only save the model it-self
                torch.save(model_to_save.state_dict(), output_model_file)
                torch.save(encoder_to_save.state_dict(), output_encoder_file)
                with open(output_config_file, 'w') as f:
                    f.write(model_to_save.config.to_json_string())
                label_map = {i: label for i, label in enumerate(label_list, 1)}
                model_config = {"bert_model": args.bert_model, "do_lower": args.do_lower_case,
                                "max_seq_length": args.max_seq_length, "num_labels": len(label_list) + 1,
                                "label_map": label_map}
                json.dump(model_config, open(os.path.join(args.output_dir, "model_config.json"), "w"))
                max_dev_f1 = F_score_dev
                best_dev_epoch = train_idx
            if F_score_test > max_test_f1:
                max_test_f1 = F_score_test
                best_test_epoch = train_idx

    print("**************************************************")
    print("The best epoch on the dev set: ", best_dev_epoch)
    print("The best Micro-F1 score on the dev set: ", max_dev_f1)
    print("The best epoch on the test set: ", best_test_epoch)
    print("The best Micro-F1 score on the test set: ", max_test_f1)
    print('\n')

    config = BertConfig(output_config_file)
    if args.mm_model == 'MTCCMBert':
        model = MTCCMBertForMMTokenClassificationCRF(config,args.use_roberta, layer_num1=args.layer_num1, layer_num2=args.layer_num2,
                                            layer_num3=args.layer_num3, num_labels=num_labels, auxnum_labels = auxnum_labels)

    else:
        print('please define your MNER Model')

    model.load_state_dict(torch.load(output_model_file))
    model.to(device)
    encoder_state_dict = torch.load(output_encoder_file)
    encoder.load_state_dict(encoder_state_dict)
    encoder.to(device)

    if args.do_eval and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        eval_examples = processor.get_test_examples(args.data_dir, args.image_filename, args.path_image)
        eval_features = convert_mm_examples_to_features(
            eval_examples, label_list, auxlabel_list, args.max_seq_length, tokenizer, args.crop_size, args.path_image)
        logger.info("***** Running Test Evaluation with the Best Model on the Dev Set*****")
        logger.info("  Num examples = %d", len(eval_examples))
        logger.info("  Batch size = %d", args.eval_batch_size)
        all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
        all_added_input_mask = torch.tensor([f.added_input_mask for f in eval_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
        all_img_feats = torch.stack([f.img_feat for f in eval_features])
        all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)
        all_auxlabel_ids = torch.tensor([f.auxlabel_id for f in eval_features], dtype=torch.long)
        all_imagelabel = torch.tensor([f.imagelabel for f in eval_features], dtype=torch.float)

        eval_data = TensorDataset(all_input_ids, all_input_mask, all_added_input_mask, all_segment_ids, all_img_feats,
                                  all_label_ids, all_auxlabel_ids,all_imagelabel)
        # Run prediction for full data
        eval_sampler = SequentialSampler(eval_data)
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)
        model.eval()
        encoder.eval()
        eval_loss, eval_accuracy = 0, 0
        nb_eval_steps, nb_eval_examples = 0, 0
        y_true = []
        y_pred = []
        y_true_idx = []
        y_pred_idx = []
        label_map = {i : label for i, label in enumerate(label_list,1)}
        label_map[0] = "PAD"
        for input_ids, input_mask, added_input_mask, segment_ids, img_feats, label_ids, auxlabel_ids,imagelabel in tqdm(eval_dataloader, desc="Evaluating"):
            input_ids = input_ids.to(device)
            input_mask = input_mask.to(device)
            added_input_mask = added_input_mask.to(device)
            segment_ids = segment_ids.to(device)
            img_feats = img_feats.to(device)
            label_ids = label_ids.to(device)
            auxlabel_ids = auxlabel_ids.to(device)
            trans_matrix = torch.tensor(trans_matrix).to(device)

            with torch.no_grad():
                imgs_f, img_mean, img_att = encoder(img_feats)
                predicted_label_seq_ids  = model(input_ids, segment_ids, input_mask, added_input_mask, img_att, trans_matrix,imagelabel,args.alpha, args.beta)
            
            logits = predicted_label_seq_ids
            label_ids = label_ids.to('cpu').numpy()
            input_mask = input_mask.to('cpu').numpy()
            for i,mask in enumerate(input_mask):
                temp_1 = []
                temp_2 = []
                tmp1_idx = []
                tmp2_idx = []
                for j, m in enumerate(mask):
                    if j == 0:
                        continue
                    if m:
                        if label_map[label_ids[i][j]] != "X" and label_map[label_ids[i][j]] != "</s>":
                            temp_1.append(label_map[label_ids[i][j]])
                            tmp1_idx.append(label_ids[i][j])
                            temp_2.append(label_map[logits[i][j]])
                            tmp2_idx.append(logits[i][j])
                    else:
                        break
                y_true.append(temp_1)
                y_pred.append(temp_2)
                y_true_idx.append(tmp1_idx)
                y_pred_idx.append(tmp2_idx)

        sentence_list = []
        test_data, imgs, _,_ = processor._read_mmtsv(os.path.join(args.data_dir, "test.txt"),args.image_filename, args.path_image)
        output_pred_file = os.path.join(args.output_dir, "mtmner_pred.txt")
        fout = open(output_pred_file, 'w', encoding='UTF-8')
        for i in range(len(y_pred)):
            sentence = test_data[i][0]
            sentence_list.append(sentence)
            img = imgs[i]
            samp_pred_label = y_pred[i]
            samp_true_label = y_true[i]
            fout.write(img+'\n')
            fout.write(' '.join(sentence)+'\n')
            fout.write(' '.join(samp_pred_label)+'\n')
            fout.write(' '.join(samp_true_label)+'\n'+'\n')
        fout.close()
        logger.info("***** Test Eval results *****")

        reverse_label_map = {label: i for i, label in enumerate(label_list, 1)}
        acc, f1, p, r = evaluate(y_pred_idx, y_true_idx, sentence_list, reverse_label_map)
        print("Overall: ", p, r, f1)
        per_f1, per_p, per_r = evaluate_each_class(y_pred_idx, y_true_idx, sentence_list, reverse_label_map, 'POS')
        print("Positive: ", per_p, per_r, per_f1)
        loc_f1, loc_p, loc_r = evaluate_each_class(y_pred_idx, y_true_idx, sentence_list, reverse_label_map, 'NEU')
        print("Neutral: ", loc_p, loc_r, loc_f1)
        org_f1, org_p, org_r = evaluate_each_class(y_pred_idx, y_true_idx, sentence_list, reverse_label_map, 'NEG')
        print("Negative: ", org_p, org_r, org_f1)

        output_eval_file = os.path.join(args.output_dir, "eval_results.txt")
        with open(output_eval_file, "w") as writer:
            #logger.info("\n%s", report)
            #writer.write(report)
            writer.write("Overall: " + str(p) + ' ' + str(r) + ' ' + str(f1) + '\n')
            writer.write("Positive: " + str(per_p) + ' ' + str(per_r) + ' ' + str(per_f1) + '\n')
            writer.write("Neutral: " + str(loc_p) + ' ' + str(loc_r) + ' ' + str(loc_f1) + '\n')
            writer.write("Negative: " + str(org_p) + ' ' + str(org_r) + ' ' + str(org_f1) + '\n')


if __name__ == "__main__":
    main()
