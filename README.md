# **Smart Cascades: An Innovative Framework for Cascading Classifiers with Comparison-driven Confidence Calibration**
## **Summary**
[1. Requirements](#1)

[2. Download weights and dataset](#2)

[3. Obtaining the performance of the cascade system on CIFAR-100-test and ImageNet-val](#3)

[4. (optional) Retraining the weights of the estimator](#4)

## <a name="1"></a>**1. Requirements**


### System Requirements
- Python 3.8 or 3.9
- Operating System: Ubuntu 18.04 or Windows 10
- GPU memory: 8GB or higher
- Disk space: 30GB or higher

###  Installation
To install the necessary packages, run the following command:

```setup
pip install -r requirements.txt
```

## <a name="2"></a>**2. Download weights and dataset**

We have uploaded the weights of the classification model, the dataset for training the estimator, and the datasets for evaluating performance of cascade system (CIFAR-100-test, ImageNet-val) to [Google Drive](https://drive.google.com/file/d/1GQQWkh8d7HU-wT9GE-1F_9D2MFJRZk_y/view?usp=share_link). Please download and unzip them to the root directory of the repository, replacing the folders with the same names.

## <a name="3"></a>**3. Obtaining the performance of the cascade system on CIFAR-100-test and ImageNet-val**

If you want to obtain the performance of a cascade system composed of any two classification models (Tab.4) that appeared in the paper, on CIFAR-100, assuming the fast model is resnet18 and the expensive model is resnet101, you can enter the following command:
```python
cd <root of the repo>
python ./scripts/script_cascade_inference_cifar100.py
--fast-model resnet18 
--fast-model-weight ./classification_model_weights/CIFAR100/resnet18.pth.tar
--exp-model resnet101 
--exp-model-weight ./classification_model_weights/CIFAR100/resnet101.pth.tar 
--estimator-weight-folder ./estimator_weights/weights_resnet18_resnet101
```
Please select and copy the names of the fast-model and exp-model from the following **name list**:
```name list<a name="namelist"></a>
Name list:
convnext_base
convnext_small
convnext_tiny
dm_nfnet_f0
mobilenetv3_large_100
mobilenetv3_small_100
resnet18
resnet34
resnet50
resnet101
tf_efficientnet_b0
tf_efficientnet_b1
vgg11_bn
volo_d1_224
volo_d2_224
volo_d3_224
```
If you want to test on ImageNet-val, you simply need to replace script_cascade_inference_cifar100.py with script_cascade_inference_ImageNet.py. For example, if we use EfficientNet-B0 as the fast model and ResNet101 as the expensive model:
```python
cd <root of the repo>
python ./scripts/script_cascade_inference_ImageNet.py
--fast-model tf_efficientnet_b0
--fast-model-weight ./classification_model_weights/ImageNet/tf_efficientnet_b0.pth.tar
--exp-model resnet101
--exp-model-weight ./classification_model_weights/ImageNet/resnet101.pth.tar 
--estimator-weight-folder ./estimator_weights/weights_ImageNet_tf_efficientnet_b0_ImageNet_resnet101_ImageNet
```
It should also be noted that the model name should be copied from the [name list](#namelist), and we have also changed the weight paths of the classification models and the estimator.
### Output
The output result will be saved as a CSV file in the root directory of the repository, with the following columns:
```dataset format
threshold,total_num,correct_num,accuracy,deep_call,use_deep
...
```
- "correct_num": the number of samples classified correctly by the cascade inference system
- "accuracy": the accuracy of the cascade inference system in CIFAR-100-test or ImageNet-val
- "deep_call": the number of samples classified by the expensive model
- "use_deep": the number of samples for which the cascade system finally used the classification result of the expensive model

P.S. If you see 'TSP' in the code, please ignore it. It was early name of our research project.
## <a name="4"></a>**4. (optional) Retraining the weights of the estimator**

In the previous section, we used pre-trained estimator weights. You can also train the estimator weights yourself. If you want to train on CIFAR100, please use `./SmartCascade_train_tabnet_cifar100.ipynb`. If you want to train on ImageNet, please use `./SmartCascade_train_tabnet_ImageNet.ipynb`. You can set the corresponding hyperparameters at the top of the ipynb file.
