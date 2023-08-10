# Llama 2

We are unlocking the power of large language models. Our latest version of Llama is now accessible to individuals, creators, researchers and businesses of all sizes so that they can experiment, innovate and scale their ideas responsibly. 

This release includes model weights and starting code for pretrained and fine-tuned Llama language models — ranging from 7B to 70B parameters.

This repository is intended as a minimal example to load [Llama 2](https://ai.meta.com/research/publications/llama-2-open-foundation-and-fine-tuned-chat-models/) models and run inference. For more detailed examples leveraging HuggingFace, see [llama-recipes](https://github.com/facebookresearch/llama-recipes/).


## Download

In order to download the model weights and tokenizer, please visit the [Meta AI website](https://ai.meta.com/resources/models-and-libraries/llama-downloads/) and accept our License.

Once your request is approved, you will receive a signed URL over email. Then run the download.sh script, passing the URL provided when prompted to start the download. Make sure that you copy the URL text itself, **do not use the 'Copy link address' option** when you right click the URL. If the copied URL text starts with: https://download.llamameta.net, you copied it correctly. If the copied URL text starts with: https://l.facebook.com, you copied it the wrong way.

Pre-requisites: make sure you have `wget` and `md5sum` installed. Then to run the script: `./download.sh`.
