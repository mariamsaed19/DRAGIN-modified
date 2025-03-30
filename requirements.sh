#!/bin/bash
pip install torch==2.1.1
pip install -r requirements.txt
python3 -m spacy download en_core_web_sm
pip install huggingface_hub==0.23.2
pip install -U beir
pip install -U bitsandbytes
pip install whoosh
pip install camel_tools
# camel_data -i defaults
pip install -U transformers
