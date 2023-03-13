import torch
import pandas as pd
from PIL import Image
import os
import numpy as np
import timm
from timm.data.transforms_factory import transforms_imagenet_eval
from timm.data.parsers.parser_image_folder import find_images_and_targets
from tqdm import tqdm
import warnings
import argparse

from dataset import create_dataset as tsp_create_dataset

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()

# base argument
parser.add_argument('dataset_path', type=str,
                    help='Path of CAFIR-100 to generate classification result dataset for training.')
parser.add_argument('json_path', type=str,
                    help='Path of json file of CAFIR-100 including indexs of images we want use.')
parser.add_argument('--output', type=str, default=os.path.join(os.path.dirname(os.path.realpath(__file__))),
                    help='Path of saving generated dataset.')
parser.add_argument('--model', type=str, default='tf_efficientnet_b4',
                    help='The classification model name for generating dataset.')
parser.add_argument('--num_classes', type=int, default=100,
                    help='Number of classification.')
parser.add_argument('--model-weight', type=str, default=None,
                    help='Weight of classification model to load.')
parser.add_argument('--batch-size', type=int, default=4,
                    help='Inference batch size.')

cf = parser.parse_args()

'''
output format:
       [filename],[correctness (1 or 0)],[index of groundtruth class],[n-dim(ImageNet:n=1000) classifcation probilities (start from 0)]
       [filename],[correctness (1 or 0)],[index of groundtruth class],[n-dim(ImageNet:n=1000) classifcation probilities (start from 0)]
        ...
'''


def model_inference(img_list, model1):


    img1 = torch.stack(img_list, dim=0)

    # Model gets predictions
    predict_result1 = model1(img1.cuda())

    # Define softmax
    sm = torch.nn.Softmax(dim=1)

    predict_result1 = sm(predict_result1)

    return predict_result1


if __name__ == '__main__':

    if cf.model_weight is not None:
        model = timm.create_model(cf.model,  num_classes=cf.num_classes, checkpoint_path=cf.model_weight)
    else:
        model = timm.create_model(cf.model, pretrained=True)

    data_config = model.default_cfg
    model.eval()
    model.cuda()

    dataset = tsp_create_dataset(
    name='cifar100_train',
    root=cf.dataset_path, #"C:\Users\yaoching\Desktop\TSP\dataset\origin\cifar100"
    tsp_data_info=cf.json_path, #'C:\Users\yaoching\Desktop\TSP\POC\json_split_for_estimator_train.json',
    split='validation',
    )

    # The transform used when classification model inferring
    cls_transform = transforms_imagenet_eval(
        img_size=data_config['input_size'][-2:],
        interpolation=data_config['interpolation'],
        mean=(0.5070751592371323,0.48654887331495095,0.4409178433670343),
        std=(0.2673342858792401,0.2564384629170883,0.27615047132568404),
        crop_pct=data_config['crop_pct']
        )
    dataset.dataset.transform=cls_transform

    cls_result = pd.DataFrame(columns=['index_in_dataset', 'correctness', 'correct_cls_index'] + ['prob_index_'+str(i) for i in range(cf.num_classes)])

    image_list = []
    index_list = []
    class_id = []

    for index in tqdm(range(len(dataset))):

        # imagenet exists grayscale images
        image,label,_ = dataset[index]
        image_list.append(image)

        real_index = str(dataset.indices[index])
        index_list.append(real_index)

        # id of groundTruth
        class_id.append(label)

        if(len(image_list) == cf.batch_size):
            predict = model_inference(image_list, model)

            image_list = []

            predict = predict.cpu().detach().numpy()  # .tolist()

            # whether the model predicts correctly
            correctness = pd.DataFrame(np.argmax(predict, axis=1) == class_id)  # [True,False,...]

            predict = pd.DataFrame(predict)
            predict = pd.concat([pd.DataFrame(index_list), correctness, pd.DataFrame(class_id), predict], axis=1, ignore_index=True)
            predict.iloc[:, 1] = predict.iloc[:, 1].map({True: 1, False: 0})
            predict.columns = ['index_in_dataset', 'correctness', 'correct_cls_index'] + ['prob_index_'+str(i) for i in range(cf.num_classes)]

            cls_result = cls_result.append(predict, ignore_index=True)
            index_list = []
            class_id = []
        
    print('DONE')

    cls_result.to_csv(os.path.join(cf.output, cf.model+'_dataset.csv'), index=False, header=True, encoding="utf_8")
