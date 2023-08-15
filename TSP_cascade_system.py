import timm
from timm.data.transforms_factory import transforms_imagenet_eval
from timm.data.parsers.parser_image_folder import find_images_and_targets

import torch
import torch.nn.functional as F
from pytorch_tabnet.tab_model import TabNetRegressor

from PIL import Image
from tqdm import tqdm

import argparse
import os
import logging
import numpy as np
import pandas as pd


class TSP_cascade_system(object):

    def __init__(self, shallow_model, shallow_model_transform, deep_model, deep_model_transform, estimator, threshold):
        """ 
        cascade system of TSP

        Args:
            estimator: the estimator should be statisfied this requirement:
                have a method predict(input) and return score:
                    input : torch.tensor with shape (batch, cls dim (like imagenet will be 1000))
                    score : torch.tensor (batch, 1)
        """
        self.shallow_model = shallow_model.cuda()
        self.shallow_model_transform = shallow_model_transform
        self.deep_model = deep_model.cuda()
        self.deep_model_transform = deep_model_transform
        self.estimator = estimator
        self.threshold = threshold

        # count how many times the deep model be called
        self.deep_call_count = 0
        self.count = 0
        self.use_deep = 0

    def cascade_inference(self, image):
        """ 
        cascade inference with single image under self.threshold 

        Args:
            image: a PIL.Image image instance

        Returns:
            torch.Tensor with shape (1, num_classes)
        """

        self.count += 1
        shallow_cls_result = self.cls_model_inference(model_type='shallow', image=image)
        confidence_s = self.estimator.predict(shallow_cls_result.detach().cpu().numpy())[0, 0]

        if confidence_s > self.threshold:
            return shallow_cls_result
        else:
            self.deep_call_count += 1
            deep_cls_result = self.cls_model_inference(model_type='deep', image=image)
            confidence_d = self.estimator.predict(deep_cls_result.detach().cpu().numpy())[0, 0]

        if confidence_s > confidence_d:
            return shallow_cls_result
        else:
            self.use_deep += 1
            return deep_cls_result

    def cascade_inference_with_batch(self):
        # TODO cascade system infer with batch
        pass

    def cls_model_inference(self, model_type, image):
        if model_type == 'shallow':
            cls_model = self.shallow_model
            transform = self.shallow_model_transform
        else:
            cls_model = self.deep_model
            transform = self.deep_model_transform

        # infer with float32
        img_transfromed = transform(image.copy()).unsqueeze(0).type(torch.FloatTensor)
        cls_result = cls_model(img_transfromed.cuda())
        cls_result = F.softmax(cls_result, dim=1)

        return cls_result

    def CAFIR_threshold_grid_search(self, dataset, search_scope, no_cascade_head):

        from dataset_utils.dataset import create_dataset as tsp_create_dataset

        # if json is provided, means will search val part of training of estimator split json
        if cf.json is not None:
            dataset = tsp_create_dataset(
                name='cifar100_train',
                root=dataset,
                tsp_data_info=cf.json,
                split='validation',
            )
        else:
            dataset = tsp_create_dataset(
                name='cifar100_test',
                root=dataset,
                tsp_data_info=None,
                split='validation',
            )

        # inference once for reuse later
        print(f'pre classify for searching threshold...')
        shallow_cls_result = []
        deep_cls_result = []

        # Load already computed result for speeding up
        if os.path.exists(os.path.join(root_path,'classification_result',f'{cf.shallow_model}_result.npy')) and os.path.exists(os.path.join(root_path,'classification_result',f'{cf.deep_model}_result.npy')): 
            print('Classification result alreay exist, will be loaded.')
            shallow_cls_result = np.load(os.path.join(root_path,'classification_result',f'{cf.shallow_model}_result.npy'))
            deep_cls_result = np.load(os.path.join(root_path,'classification_result',f'{cf.deep_model}_result.npy'))
        else: 
            for i in tqdm(range(len(dataset))):
                shallow_cls_result.append(self.cls_model_inference(model_type='shallow', image=dataset[i][0]).detach().cpu().numpy())
                deep_cls_result.append(self.cls_model_inference(model_type='deep', image=dataset[i][0]).detach().cpu().numpy())

            shallow_cls_result = np.concatenate(shallow_cls_result, axis=0)
            deep_cls_result = np.concatenate(deep_cls_result, axis=0)
            np.save(os.path.join(root_path,'classification_result',f'{cf.shallow_model}_result.npy'),shallow_cls_result)
            np.save(os.path.join(root_path,'classification_result',f'{cf.deep_model}_result.npy'),deep_cls_result)

        # inference once for reuse later
        print(f'pre compute confidence for searching threshold...')
        shallow_cls_result_confidence = []
        deep_cls_result_confidence = []

        for i in tqdm(range(len(dataset))):
            shallow_cls_result_confidence.append(self.estimator.predict(shallow_cls_result[[i]])[0, 0])
            deep_cls_result_confidence.append(self.estimator.predict(deep_cls_result[[i]])[0, 0])

        output = pd.DataFrame(columns=['threshold', 'total_num', 'correct_num', 'accuracy', 'deep_call', 'use_deep'])

        for t in search_scope:

            self.deep_call_count = 0
            self.correct = 0
            self.threshold = t
            self.use_deep = 0
            for i in tqdm(range(shallow_cls_result.shape[0])):
                confidence_s = shallow_cls_result_confidence[i]  # np.max(shallow_cls_result[[i]])

                if confidence_s > self.threshold:
                    result = shallow_cls_result[i]

                elif no_cascade_head == False:
                    '''
                    self.deep_call_count += 1
                    result = deep_cls_result[i]
                    self.use_deep+=1
                    '''
                    self.deep_call_count += 1
                    confidence_d = deep_cls_result_confidence[i]

                    if confidence_s > confidence_d:
                        result = shallow_cls_result[i]
                    else:
                        result = deep_cls_result[i]
                        self.use_deep += 1
                elif no_cascade_head == True:
                    self.deep_call_count += 1
                    result = deep_cls_result[i]
                    self.use_deep += 1
                # result = deep_cls_result[i] + shallow_cls_result[i]
                if np.argmax(result) == dataset[i][1]:
                    self.correct += 1

            logger.info(f'threshold:{self.threshold} total_num:{len(dataset)} correct_num:{self.correct} accuracy:{self.correct/len(dataset)} deep_call:{self.deep_call_count} use_deep:{self.use_deep}')
            output = output.append({'threshold': self.threshold, 'total_num': len(dataset), 'correct_num': self.correct,
                                   'accuracy': self.correct/len(dataset), 'deep_call': self.deep_call_count, 'use_deep': self.use_deep},ignore_index=True)
        if no_cascade_head == True:
            output.to_csv(os.path.join(root_path,f'{cf.shallow_model}_{cf.deep_model}_{os.path.splitext(os.path.split(cf.estimator_weight)[-1])[0]}_No_Head.csv'),index=False)
        else:
            output.to_csv(os.path.join(root_path,f'{cf.shallow_model}_{cf.deep_model}_{os.path.splitext(os.path.split(cf.estimator_weight)[-1])[0]}.csv'),index=False)

    def LTC_CAFIR_threshold_grid_search(self, dataset, search_scope):

        from dataset_utils.dataset import create_dataset as tsp_create_dataset

        # if json is provided, means will search val part of training of estimator split json
        if cf.json is not None:
            dataset = tsp_create_dataset(
                name='cifar100_train',
                root=dataset,
                tsp_data_info=cf.json,
                split='validation',
            )
        else:
            dataset = tsp_create_dataset(
                name='cifar100_test',
                root=dataset,
                tsp_data_info=None,
                split='validation',
            )

        # inference once for reuse later
        print(f'pre classify for searching threshold...')
        shallow_cls_result = []
        deep_cls_result = []

        tmp_fast_name = os.path.split(cf.shallow_model_weight)[-1].split('.')[0]

        # Load already computed result for speeding up
        if os.path.exists(f"./classification_result/{cf.shallow_model}_{cf.deep_model}_{tmp_fast_name}_result.npy"):
            print('Classification result of fast model alreay exist, will be loaded.')
            shallow_cls_result = np.load(f"./classification_result/{cf.shallow_model}_{cf.deep_model}_{tmp_fast_name}_result.npy")
        else: 
            for i in tqdm(range(len(dataset))):
                shallow_cls_result.append(self.cls_model_inference(model_type='shallow', image=dataset[i][0]).detach().cpu().numpy())

            shallow_cls_result = np.concatenate(shallow_cls_result, axis=0)
            np.save(f"./classification_result/{cf.shallow_model}_{cf.deep_model}_{tmp_fast_name}_result.npy",shallow_cls_result)

        # Load already computed result for speeding up
        if os.path.exists(f"./classification_result/{cf.deep_model}_result.npy"):
            print('Classification result of exp model alreay exist, will be loaded.')
            deep_cls_result = np.load(f"./classification_result/{cf.deep_model}_result.npy")
        else: 
            for i in tqdm(range(len(dataset))):
                deep_cls_result.append(self.cls_model_inference(model_type='deep', image=dataset[i][0]).detach().cpu().numpy())

            deep_cls_result = np.concatenate(deep_cls_result, axis=0)
            np.save(f"./classification_result/{cf.deep_model}_result.npy",deep_cls_result)


        output = pd.DataFrame(columns=['threshold', 'total_num', 'correct_num', 'accuracy', 'deep_call', 'use_deep'])

        for t in search_scope:

            self.deep_call_count = 0
            self.correct = 0
            self.threshold = t
            self.use_deep = 0
            for i in tqdm(range(shallow_cls_result.shape[0])):
                confidence_s = np.max(shallow_cls_result[[i]])

                if confidence_s > self.threshold:
                    result = shallow_cls_result[i]
                else:
                    self.deep_call_count += 1
                    result = deep_cls_result[i]
                    self.use_deep += 1

                if np.argmax(result) == dataset[i][1]:
                    self.correct += 1

            logger.info(f'threshold:{self.threshold} total_num:{len(dataset)} correct_num:{self.correct} accuracy:{self.correct/len(dataset)} deep_call:{self.deep_call_count} use_deep:{self.use_deep}')
            output = output.append({'threshold': self.threshold, 'total_num': len(dataset), 'correct_num': self.correct,
                        'accuracy': self.correct/len(dataset), 'deep_call': self.deep_call_count, 'use_deep': self.use_deep},ignore_index=True)

        output.to_csv(f'./LTC_{cf.shallow_model}_{cf.deep_model}_{tmp_fast_name}.csv',index=False)
            
    def IMAGENET_threshold_grid_search(self, dataset, search_scope):

        samples, class_to_idx = find_images_and_targets(folder=cf.data_dir, class_to_idx=None)

        # inference once for reuse later
        print(f'pre classify for searching threshold...')
        shallow_cls_result = []
        deep_cls_result = []

        # Load already computed result for speeding up
        if os.path.exists(os.path.join(root_path,'classification_result',f"{cf.shallow_model}_ImageNet_result.npy")):
            print('Classification result alreay exist, will be loaded.')
            shallow_cls_result = np.load(os.path.join(root_path,'classification_result',f"{cf.shallow_model}_ImageNet_result.npy"))
        else: 
            for i in tqdm(range(len(samples))):
                image = Image.open(samples[i][0]).convert("RGB")
                shallow_cls_result.append(self.cls_model_inference(model_type='shallow', image=image).detach().cpu().numpy())

            shallow_cls_result = np.concatenate(shallow_cls_result, axis=0)
            np.save(os.path.join(root_path,'classification_result',f"{cf.shallow_model}_ImageNet_result.npy"),shallow_cls_result)

        # Load already computed result for speeding up
        if os.path.exists(os.path.join(root_path,'classification_result',f"{cf.deep_model}_ImageNet_result.npy")):
            print('Classification result alreay exist, will be loaded.')
            deep_cls_result = np.load(os.path.join(root_path,'classification_result',f"{cf.deep_model}_ImageNet_result.npy"))
        else: 
            for i in tqdm(range(len(samples))):
                image = Image.open(samples[i][0]).convert("RGB")
                deep_cls_result.append(self.cls_model_inference(model_type='deep', image=image).detach().cpu().numpy())

            deep_cls_result = np.concatenate(deep_cls_result, axis=0)
            np.save(os.path.join(root_path,'classification_result',f"{cf.deep_model}_ImageNet_result.npy"),deep_cls_result)

        # inference once for reuse later
        print(f'pre compute confidence for searching threshold...')
        shallow_cls_result_confidence = []
        deep_cls_result_confidence = []

        estimator_batch = 256
        count = 0
        for i in tqdm(range(len(samples))):

            if i % estimator_batch == 0:
                tmp_s_c = self.estimator.predict(shallow_cls_result[count:count+estimator_batch])
                tmp_d_c = self.estimator.predict(deep_cls_result[count:count+estimator_batch])
                count += estimator_batch

                shallow_cls_result_confidence += tmp_s_c[:,0].tolist()
                deep_cls_result_confidence += tmp_d_c[:,0].tolist()

            elif ((len(samples)-1) -count)<estimator_batch:
                tmp_s_c = self.estimator.predict(shallow_cls_result[count:])
                tmp_d_c = self.estimator.predict(deep_cls_result[count:])
                count += estimator_batch

                shallow_cls_result_confidence += tmp_s_c[:,0].tolist()
                deep_cls_result_confidence += tmp_d_c[:,0].tolist()
                break

        output = pd.DataFrame(columns=['threshold', 'total_num', 'correct_num', 'accuracy', 'deep_call', 'use_deep'])

        for t in search_scope:

            self.deep_call_count = 0
            self.correct = 0
            self.threshold = t
            self.use_deep = 0

            for i in tqdm(range(shallow_cls_result.shape[0])):
                confidence_s = shallow_cls_result_confidence[i]  # np.max(shallow_cls_result[[i]])

                if confidence_s > self.threshold:
                    result = shallow_cls_result[i]
                else:
                    '''
                    self.deep_call_count += 1
                    result = deep_cls_result[i]
                    self.use_deep +=1
                    '''
                    self.deep_call_count += 1
                    confidence_d = deep_cls_result_confidence[i]

                    if confidence_s > confidence_d:
                        result = shallow_cls_result[i]
                    else:
                        result = deep_cls_result[i]
                        self.use_deep += 1

                if np.argmax(result) == samples[i][1]:
                    self.correct += 1

            output = output.append({'threshold': self.threshold, 'total_num': len(samples), 'correct_num': self.correct,
                        'accuracy': self.correct/len(samples), 'deep_call': self.deep_call_count, 'use_deep': self.use_deep},ignore_index=True)

            logger.info(f'threshold:{self.threshold} total_num:{len(samples)} correct_num:{self.correct} accuracy:{self.correct/len(samples)} deep_call:{self.deep_call_count} use_deep:{self.use_deep}')
        output.to_csv(os.path.join(root_path,f'IMAGENET_{cf.shallow_model}_{cf.deep_model}_{os.path.splitext(os.path.split(cf.estimator_weight)[-1])[0]}.csv'),index=False)

    def LTC_IMAGENET_threshold_grid_search(self, dataset, search_scope):

        samples, class_to_idx = find_images_and_targets(folder=cf.data_dir, class_to_idx=None)

        # inference once for reuse later
        print(f'pre classify for searching threshold...')
        shallow_cls_result = []
        deep_cls_result = []

        tmp_fast_name = os.path.split(cf.shallow_model_weight)[-1].split('.')[0]

        # Load already computed result for speeding up
        if os.path.exists(f"./classification_result/LTC_{cf.shallow_model}_{cf.deep_model}_{tmp_fast_name}_ImageNet_result.npy"):
            print('Classification result alreay exist, will be loaded.')
            shallow_cls_result = np.load(f"./classification_result/LTC_{cf.shallow_model}_{cf.deep_model}_{tmp_fast_name}_ImageNet_result.npy")
        else: 
            for i in tqdm(range(len(samples))):
                image = Image.open(samples[i][0]).convert("RGB")
                shallow_cls_result.append(self.cls_model_inference(model_type='shallow', image=image).detach().cpu().numpy())

            shallow_cls_result = np.concatenate(shallow_cls_result, axis=0)
            np.save(f"./classification_result/LTC_{cf.shallow_model}_{cf.deep_model}_{tmp_fast_name}_ImageNet_result.npy",shallow_cls_result)

        # Load already computed result for speeding up
        if os.path.exists(f"./classification_result/{cf.deep_model}_ImageNet_result.npy"):
            print('Classification result alreay exist, will be loaded.')
            deep_cls_result = np.load(f"./classification_result/{cf.deep_model}_ImageNet_result.npy")
        else: 
            for i in tqdm(range(len(samples))):
                image = Image.open(samples[i][0]).convert("RGB")
                deep_cls_result.append(self.cls_model_inference(model_type='deep', image=image).detach().cpu().numpy())

            deep_cls_result = np.concatenate(deep_cls_result, axis=0)
            np.save(f"./classification_result/{cf.deep_model}_ImageNet_result.npy",deep_cls_result)

        output = pd.DataFrame(columns=['threshold', 'total_num', 'correct_num', 'accuracy', 'deep_call', 'use_deep'])

        for t in search_scope:

            self.deep_call_count = 0
            self.correct = 0
            self.threshold = t
            self.use_deep = 0

            for i in tqdm(range(shallow_cls_result.shape[0])):
                confidence_s = np.max(shallow_cls_result[[i]])

                if confidence_s > self.threshold:
                    result = shallow_cls_result[i]
                else:
                    self.deep_call_count += 1
                    result = deep_cls_result[i]
                    self.use_deep +=1
                    

                if np.argmax(result) == samples[i][1]:
                    self.correct += 1

            output = output.append({'threshold': self.threshold, 'total_num': len(samples), 'correct_num': self.correct,
                        'accuracy': self.correct/len(samples), 'deep_call': self.deep_call_count, 'use_deep': self.use_deep},ignore_index=True)

            logger.info(f'threshold:{self.threshold} total_num:{len(samples)} correct_num:{self.correct} accuracy:{self.correct/len(samples)} deep_call:{self.deep_call_count} use_deep:{self.use_deep}')
        

        output.to_csv(f'./LTC_IMAGENET_{cf.shallow_model}_{cf.deep_model}_{tmp_fast_name}.csv',index=False)

parser = argparse.ArgumentParser()

# base argument
parser.add_argument('data_dir', metavar='DIR',
                    help='path to dataset to evalutate cascade system, it should be follow imagefolder structure \
                        (https://pytorch.org/vision/stable/generated/torchvision.datasets.ImageFolder.html)')
parser.add_argument('dataset', type=str, default='ImageNet',
                    help='ImageNet or CAFIR-100, the program will use corresponding dataset class.')
parser.add_argument('--shallow-model', type=str, default='tf_efficientnet_b0',
                    help='Shallow model of cascade system.')
parser.add_argument('--shallow-model-weight', type=str, default=None,
                    help='Path of weight of shallow model of cascade system.')
parser.add_argument('--shallow-model-FLOPs', type=float, default=0.39,
                    help='FLOPs of shallow model of cascade system.')
parser.add_argument('--deep-model', type=str, default='tf_efficientnet_b4',
                    help='Deep model of cascade system.')
parser.add_argument('--deep-model-weight', type=str, default=None,
                    help='Path of weight of deep model of cascade system.')
parser.add_argument('--deep-model-FLOPs', type=float, default=4.2,
                    help='FLOPs of deep model of cascade system.')
parser.add_argument('--num-classes', type=int, default=1000,
                    help='Number of label classes.')
parser.add_argument('--estimator', type=str, default='TabNet',
                    help='The estimator will be used in cascade system.')
parser.add_argument('--estimator-weight', type=str, default='./TSP_tabnet_model.zip',
                    help='The path of weight of estimator will be used in cascade system.')
parser.add_argument('--threshold', type=float, default=-999,
                    help='Calculate the FLOPs and accuracy of the system under this threshold.')
parser.add_argument('--search', action="store_true", default=False,
                    help='Sorting classification result from largest to smallest.')
parser.add_argument('--log', type=str, default=os.path.join(os.path.dirname(os.path.realpath(__file__)), 'log.txt'),
                    help='Path of saving log.')
parser.add_argument('--performance-result', type=str, default=os.path.join(os.path.dirname(os.path.realpath(__file__)), 'performance-result.csv'),
                    help='The performance of the system under different thresholds.')
parser.add_argument('--ltc', action='store_true', default=False,
                    help='Use ltc cascade system.')
parser.add_argument('--json', type=str, default=None,
                    help='If provide, means will search in val part of train, not test.')
parser.add_argument('--no-cascade-head', action="store_true", default=False,
                    help='Whether not use cascade head.')

cf = parser.parse_args()

# The path where the output log is saved
log_path = cf.log
logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)
handler = logging.FileHandler(log_path)
handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

root_path = os.path.dirname(os.path.abspath(__file__))

os.path.exists(os.path.join(root_path,'classification_result'))
if not os.path.exists(os.path.join(root_path,'classification_result')):                  
    os.makedirs(os.path.join(root_path,'classification_result')) 
            
if __name__ == '__main__':

    shallow_model = cf.shallow_model
    deep_model = cf.deep_model

    if cf.dataset == 'ImageNet':
        num_class = 1000
    elif cf.dataset == 'CAFIR-100':
        num_class = 100
    else:
        raise Exception('dataset type ilegal !')

    # define shallow model
    if cf.shallow_model_weight is not None:
        model_s = timm.create_model(shallow_model, checkpoint_path=cf.shallow_model_weight, num_classes=num_class)
    else:
        model_s = timm.create_model(shallow_model, pretrained=True)

    data_config_s = model_s.default_cfg
    model_s.cuda()
    model_s.eval()
    print('shallow model config: ', data_config_s)

    if cf.dataset == 'ImageNet':
        mean = data_config_s['mean']
        std = data_config_s['std']
    elif cf.dataset == 'CAFIR-100':
        mean = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
        std = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)

    # transform of shallow model
    shallow_model_transform = transforms_imagenet_eval(
        img_size=data_config_s['input_size'][-2:],
        interpolation=data_config_s['interpolation'],
        mean=mean,
        std=std,
        crop_pct=data_config_s['crop_pct'],)

    # define deep model
    if cf.deep_model_weight is not None:
        model_d = timm.create_model(deep_model, checkpoint_path=cf.deep_model_weight, num_classes=num_class)
    else:
        model_d = timm.create_model(deep_model, pretrained=True)
    data_config_d = model_d.default_cfg
    model_d.cuda()
    model_d.eval()
    print('deep model config: ', data_config_d)

    # transform of deep model
    deep_model_transform = transforms_imagenet_eval(
        img_size=data_config_d['input_size'][-2:],
        interpolation=data_config_d['interpolation'],
        mean=mean,
        std=std,
        crop_pct=data_config_d['crop_pct'],)
    
    if cf.ltc == False:
        # Load estimator
        if cf.estimator == 'TabNet':
            print('Estimator is TabNet!')
            estimator = TabNetRegressor()
            estimator.load_model(cf.estimator_weight)

        else:
            print('Estimator is FC!')
            from Jack.pytorch_fullyconnect.inference import *
            estimator = NeuralNet_TSP(num_class)
            estimator = get_torch_model(estimator, cf.estimator_weight)
    else: 
        # fake estimator and will not be used.
        estimator = TabNetRegressor()

    cascade_sys = TSP_cascade_system(shallow_model=model_s,
                                     shallow_model_transform=shallow_model_transform,
                                     deep_model=model_d,
                                     deep_model_transform=deep_model_transform,
                                     estimator=estimator,
                                     threshold=cf.threshold)

    if cf.dataset == 'CAFIR-100':
        if cf.search is True:
            if cf.ltc is True:
                cascade_sys.LTC_CAFIR_threshold_grid_search(cf.data_dir, search_scope=np.linspace(0, 1.1, 221))
            else:
                cascade_sys.CAFIR_threshold_grid_search(cf.data_dir, search_scope=np.linspace(0, 1.1, 221), no_cascade_head=cf.no_cascade_head)
        else:
            if cf.ltc is True:
                cascade_sys.LTC_CAFIR_threshold_grid_search(cf.data_dir, search_scope=[cf.threshold])
            else:
                cascade_sys.CAFIR_threshold_grid_search(cf.data_dir, search_scope=[cf.threshold], no_cascade_head=cf.no_cascade_head)
    else:
        if cf.search is True:
            if cf.ltc is True:
                cascade_sys.LTC_IMAGENET_threshold_grid_search(cf.data_dir, search_scope=np.linspace(0, 1.1, 221))
            else:
                cascade_sys.IMAGENET_threshold_grid_search(cf.data_dir, search_scope=np.linspace(0, 1.1, 221))
        else:
            samples, class_to_idx = find_images_and_targets(folder=cf.data_dir, class_to_idx=None)
            # Record the metrics under different threshold
            output = pd.DataFrame(columns=['threshold', 'accuracy', 'deep_call_count'])

            # Set threshold to desired value
            cascade_sys.threshold = cf.threshold
            correct_num = 0
            for i in tqdm(range(len(samples))):
                image = Image.open(samples[i][0]).convert("RGB")

                temp_result = cascade_sys.cascade_inference(image)
                _, pred_index = torch.max(temp_result, dim=1)

                if pred_index == samples[i][1]:
                    correct_num += 1

            accuracy = round(correct_num/cascade_sys.count, 4)

            print(f'threshold:{cf.threshold} total_num:{cascade_sys.count} correct_num:{correct_num} accuracy:{accuracy} deep_call:{cascade_sys.deep_call_count} use_deep:{cascade_sys.use_deep}')

            # Save in log.txt
            logger.info(f'threshold:{cf.threshold} total_num:{cascade_sys.count} correct_num:{correct_num} accuracy:{accuracy} deep_call:{cascade_sys.deep_call_count} use_deep:{cascade_sys.use_deep}')
