FROM nvcr.io/nvidia/pytorch:24.07-py3

RUN pip install \
  accelerate==1.3.0 \
  appdirs==1.4.4 \
  loralib==0.1.2 \
  bitsandbytes==0.45.2 \
  black==25.1.0 \
  black[jupyter]==25.1.0 \
  datasets==3.3.0 \
  fire==0.7.0 \
  peft==0.14.0 \
  transformers==4.48.3 \
  timm==1.0.14 \
  sentencepiece==0.2.0 \
  py7zr==0.22.0 \
  scipy==1.13.1 \
  optimum==1.24.0 \
  matplotlib==3.9.1 \
  chardet==5.2.0 \
  openai==1.63.0 \
  typing-extensions==4.12.0 \
  tabulate==0.9.0 \
  evaluate==0.4.3 \
  rouge_score==0.1.2 \
  pyyaml==6.0.1 \
  faiss-gpu==1.7.2 \
  unstructured[pdf]==0.16.20 \
  sentence_transformers==3.4.1 \
  codeshield==1.0.1 \
  gradio==5.16.0 \
  markupsafe==2.0.1 \
  wandb==0.18.7

RUN git clone --no-checkout https://github.com/meta-llama/llama-cookbook.git \
  && cd llama-cookbook \
  && git fetch --depth 1 origin faae2fd877995430906e1d0904131ecdaa89a604 \
  && git checkout faae2fd877995430906e1d0904131ecdaa89a604 \
  && pip install -U pip setuptools \
  && pip install -e .

RUN wget https://raw.githubusercontent.com/meta-llama/llama-cookbook/faae2fd877995430906e1d0904131ecdaa89a604/src/llama_cookbook/finetuning.py