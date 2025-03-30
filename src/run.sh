#!/bin/bash

echo "Start generation"
CUDA_VISIBLE_DEVICES=1 python3 main.py -c ../config/aya-expanse-quantized/ArabicaQA/DRAGIN.json 
# 1> log-aradpr-question-few6.log 2> log-aradpr-question-few6.err

