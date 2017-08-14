#Google cloud engine:

To install on tensorflow/Keras on GPU,[follow this tutorial](https://medium.com/google-cloud/running-jupyter-notebooks-on-gpu-on-google-cloud-d44f57d22dbd)

Make sure to install Cudnn 5.1 for CUDA 8.0 (when installing 8.0 like in the tutorial), since that's the one the installed tensorflow-gpu uses as of August 2017.

## connect over ssh
gcloud compute ssh disdat-gpu-1

## download files from bucket

''' gsutil -m cp -r  gs://disdat-bucket/data . '''

## start notebook 
''' jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser & '''
''' tensorboard '''
In case it doesn't work --> firewall in console