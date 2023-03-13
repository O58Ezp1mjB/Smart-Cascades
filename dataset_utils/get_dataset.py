from torchvision.datasets.utils import download_and_extract_archive, download_url
# from torchvision.datasets.imagenet import ImageNet
# from torchvision.datasets import CIFAR100
from cifar100_dataset import LTC_CIFAR100 as CIFAR100
from imagenet_dataset import LTC_ImageNet as ImageNet
import os


def _imagenet_train(root, transform=None, expensive_prediction_path=None):
    """
    Download imagenet_train dataset if not exist.

    Parameters
    ----------
    root : string
        path to store the dataset

    Returns
    -------
    imagenet_train_dataset : torchvision.datasets.imagenet
        object of imagenet_train dataset
    """
    if not os.path.isfile(os.path.join(root, "ILSVRC2012_img_train.tar")):
        download_url(
            url="https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_train.tar",
            root=root,
            md5="1d675b47d978889d74fa0da5fadfb00e",
        )
    if not os.path.isfile(os.path.join(root, "ILSVRC2012_devkit_t12.tar.gz")):
        download_url(
            url="https://image-net.org/data/ILSVRC/2012/ILSVRC2012_devkit_t12.tar.gz",
            root=root,
            md5="fa75699e90414af021442c21a62c3abf",
        )
    print("Imagenet train data already exists, and it is prepared to use.")
    imagenet_val_dataset = ImageNet(root=root, split="train", expensive_prediction_path=expensive_prediction_path)
    return imagenet_val_dataset

def _imagenet_val(root, transform=None, expensive_prediction_path=None):
    """
    Download imagenet_val dataset if not exist.

    Parameters
    ----------
    root : string
        path to store the dataset

    Returns
    -------
    imagenet_val_dataset : torchvision.datasets.imagenet
        object of imagenet_val dataset
    """
    if not os.path.isfile(os.path.join(root, "ILSVRC2012_img_val.tar")):
        download_url(
            url="https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_val.tar",
            root=root,
            md5="29b22e2961454d5413ddabcf34fc5622",
        )
    if not os.path.isfile(os.path.join(root, "ILSVRC2012_devkit_t12.tar.gz")):
        download_url(
            url="https://image-net.org/data/ILSVRC/2012/ILSVRC2012_devkit_t12.tar.gz",
            root=root,
            md5="fa75699e90414af021442c21a62c3abf",
        )
    print("Imagenet val data already exists, and it is prepared to use.")
    imagenet_val_dataset = ImageNet(root=root, split="val", expensive_prediction_path=expensive_prediction_path)
    return imagenet_val_dataset

def _cifar100_train(root, transform=None, expensive_prediction_path=None):
    """
    Download cifar100_train dataset if not exist.

    Parameters
    ----------
    root : string
        path to store the dataset

    Returns
    -------
    cifar100_train : torchvision.datasets.CIFAR100
        object of cifar100_train dataset
    """
    cifar100_train_dataset = CIFAR100(root=root, train=True, transform=transform, download=True, expensive_prediction_path=expensive_prediction_path)
    return cifar100_train_dataset

def _cifar100_val(root, transform=None, expensive_prediction_path=None):
    """
    Download cifar100_val dataset if not exist.
    
    Parameters
    ----------
    root : string
        path to store the dataset

    Returns
    -------
    cifar100_val_dataset : torchvision.datasets.CIFAR100
        object of cifar100_val  dataset
    """
    cifar100_val_dataset = CIFAR100(root=root, train=True, transform=transform, download=True, expensive_prediction_path=expensive_prediction_path)
    return cifar100_val_dataset

def _cifar100_test(root, transform=None, expensive_prediction_path=None):
    """
    Download cifar100_val dataset if not exist.
    
    Parameters
    ----------
    root : string
        path to store the dataset

    Returns
    -------
    cifar100_val_dataset : torchvision.datasets.CIFAR100
        object of cifar100_val  dataset
    """
    cifar100_val_dataset = CIFAR100(root=root, train=False, transform=transform, download=True, expensive_prediction_path=expensive_prediction_path)
    return cifar100_val_dataset


_DATASET_SYNONYM = dict(
    imagenet_train=_imagenet_train, 
    imagenet_val=_imagenet_val,
    imagenet_test=None,
    cifar100_train=_cifar100_train,
    cifar100_val=_cifar100_val,
    cifar100_test=_cifar100_test,
)

def get_dataset(name, root, expensive_prediction_path=None):
    """
        Create a pytorch dataset object that handles data download and preparation.

        Parameters
        ----------
        expensive_prediction_path: string
            path of expensive model prediction result
        name : string
            name of dataset
        root : string
            path to store the dataset
    """
    if not _DATASET_SYNONYM.__contains__(name):
        raise ValueError(f"Parameter \'name\' get unexpected \'{name}\'. It should be the following {list(_DATASET_SYNONYM.keys())}")
    # check download and get torch dataset
    return _DATASET_SYNONYM[name](root=root, transform=None, expensive_prediction_path=expensive_prediction_path)


if __name__ == '__main__':
    _dataset = get_dataset("cifar100_train", "./dataset/origin/cifar100")
    print(len(_dataset))
    