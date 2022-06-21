#!/usr/bin/env bash
for i in 'twitter2015' 'twitter2017' #  'twitter2015'
do
  echo 'run_mtmner_crf.py'
  echo ${i}
  echo ${k}
  PYTHONIOENCODING=utf-8 CUDA_VISIBLE_DEVICES=0 python run_cmmt_crf.py --task_name=${i} 
done
