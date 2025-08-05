# `llama.cpp` Serverless Inference with Knative
Knative is a super easy way to deploy serverless applications in a Kubernetes cluster.
This guide will show you how to deploy a serverless inference application with Knative on Kubernetes.

## Prerequisites
* Access to gated Llama repositories on HuggingFace
* Nebius Managed Kubernetes cluster with at least 1 GPU that has enough memory to fit the model you wish to run.
* Nebius Object Storage Bucket
* Machine with preferrably fast internet connection and `awscli` installed

## Deployment Instructions
### Part I: Preparing model weights
Before we run inference against our model, we need to convert it to GGUF and upload it to Object Storage so it can be downloaded to a Knative container.
1. Download the model from HuggingFace https://huggingface.co/docs/huggingface_hub/en/guides/download#download-an-entire-repository
2. Clone this repo locally: https://github.com/ggml-org/llama.cpp
3. Use this script to convert the model to GGUF: https://github.com/ggml-org/llama.cpp/blob/master/convert_hf_to_gguf.py
4. Make sure to rename the model to `model.gguf`
4. Follow our guide for getting started with Object Storage and uploading a file to upload the GGUF model to object storage: https://docs.nebius.com/object-storage/quickstart


### Part II: Setup Knative
To install Knative, follow their official documentation for installing Knative Serving: https://knative.dev/docs/install/yaml-install/serving/install-serving-with-yaml/
Keep in mind this project only uses Knative Serving, so it's unnecessary to install Knative Eventing.

### Part III: Configure and deploy application
Now that we've done all the necessary preparation work, we can move on to actually deploying our application as a Knative service.
1. Clone the repository that this README belongs to
2. Edit `object_storage_auth.yml` to include the necessary authentication, region, and bucket where your GGUF model is stored
3. Deploy the Knative service with `kubectl create -f llamacpp.yml`

### Part IV: Interact with llama.cpp server: 
Follow the official llama.cpp docs to interact with the server: https://llama-cpp-python.readthedocs.io/en/latest/#openai-compatible-web-server

Make sure to replace `localhost` with the endpoint URL from `kubectl get ksvc`