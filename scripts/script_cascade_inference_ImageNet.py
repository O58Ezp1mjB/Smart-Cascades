import os
import argparse

def get_project_path():

    script_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.dirname(script_dir)

root_path = get_project_path()

parser = argparse.ArgumentParser()

parser.add_argument('--dataset-path', type=str, default=os.path.join(root_path,'dataset_images','val'), help="Path of ImageNet-val dataset folder")
parser.add_argument('--fast-model', type=str, default='tf_efficientnet_b0', help="Model name of fast model in cascade inference")
parser.add_argument('--fast-model-weight', type=str, default=os.path.join(root_path,'classification_model_weights','ImageNet','tf_efficientnet_b0.pth.tar'), help="Path of fast model weight")
parser.add_argument('--exp-model', type=str, default='resnet101', help="Model name of expensive model in cascade inference")
parser.add_argument('--exp-model-weight', type=str, default=os.path.join(root_path,'classification_model_weights','ImageNet','resnet101.pth.tar'), help="Path of expensive model weight")
parser.add_argument('--estimator-weight-folder', type=str, default=os.path.join(root_path,'estimator_weights','weights_ImageNet_tf_efficientnet_b0_ImageNet_resnet101_ImageNet'), help="Path of folder that including weights of estimator of different seeds")

cf = parser.parse_args()

if __name__ == '__main__':

    # get path of program will be excecuted
    pyfile_path = os.path.join(root_path,'TSP_cascade_system.py')
    
    dataset_name = "ImageNet"

    for f in os.listdir(cf.estimator_weight_folder):

        estimator_weight = os.path.join(cf.estimator_weight_folder,f)
        command = f"python {pyfile_path} {cf.dataset_path} {dataset_name} --shallow-model {cf.fast_model} --shallow-model-weight {cf.fast_model_weight} \
                    --deep-model {cf.exp_model} --deep-model-weight {cf.exp_model_weight} --search --estimator TabNet --estimator-weight {estimator_weight}"
        print('We will run the following command:\n',command)
        os.system(command)