import os
import argparse

def get_project_path():

    script_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.dirname(script_dir)

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='cifar100_train', help="Dataset type:cifar100_train, cifar100_test, imagenet_train or imagenet_val")

cf = parser.parse_args()

if __name__ == '__main__':

    root_path = get_project_path()
    # get path of program will be excecuted
    pyfile_path = os.path.join(root_path,'dataset_utils','dataset.py')
    
    download_path = os.path.join(root_path,'dataset_images')
    json_path = os.path.join(root_path,'seeds_json','empty_for_download.json')

    command = f'python {pyfile_path} --root {download_path} --dataset {cf.dataset} --json {json_path} --generate'
    print('We will run the following command:\n',command)
    os.system(command)
    print('Please ignore above Dataset info ~')