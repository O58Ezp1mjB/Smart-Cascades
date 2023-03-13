import os
import argparse

def get_project_path():

    script_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.dirname(script_dir)

root_path = get_project_path()

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='tf_efficientnet_b0', help="Classification name to generate estimator dataset")
parser.add_argument('--model-weight', type=str, default=os.path.join(root_path,'classification_model_weights','ImageNet','tf_efficientnet_b0.pth.tar'), help="Absolute path of classification model weight")

cf = parser.parse_args()

if __name__ == '__main__':

    # get path of program will be excecuted
    pyfile_path = os.path.join(root_path,'dataset_utils','generate_dataset_ImageNet.py')
    
    dataset_path = os.path.join(root_path,'dataset_images')
    json_path = os.path.join(root_path,'seeds_json','imagenet_train_seed111.json')
    output_path = os.path.join(root_path,'dataset_estimator','ImageNet')
    command = f'python {pyfile_path} {dataset_path} {json_path} --model {cf.model} --model-weight {cf.model_weight} --output {output_path}'
    print('We will run the following command:\n',command)
    os.system(command)


