from torch.utils.data import Dataset, DataLoader
import numpy as np
from .metrics import Metric
import torch.nn as nn
import torch
from torch.nn.utils import clip_grad_norm_
from .tab_network import EmbeddingGenerator, TabNetNoEmbeddings
from .utils import define_device


class TSP_Dataset(Dataset):
    """
    The dataset will split input X.

    Parameters
    ----------
    X : 2D array
        The input matrix of classification result of shallow model and deep model (batch, 2 x classification dim)
    y : 2D array
        The correctness of classification of shallow model and deep model, and the groundtruth index of this classification (batch, 3)

    """

    def __init__(self, x, y):

        if len(x.shape) != 2 or ((x.shape[1] % 2) != 0):
            raise ValueError("The shape of X should be (batch, 2 x classification dim)!")

        half_len = int(x.shape[1] / 2)

        # Sort x from largest to smallest
        self.shallow_cls_result = x[:, :half_len]
        self.deep_cls_result = x[:, half_len:]
        self.y = y

    def __len__(self):
        return len(self.shallow_cls_result)

    def __getitem__(self, index):
        shallow_cls_result, deep_cls_result, y = self.shallow_cls_result[index], self.deep_cls_result[index], self.y[index]
        return (shallow_cls_result, deep_cls_result), y


'''
Possible replace position:
    abstract_model.py line 680 (_construct_loaders)
'''


def create_dataloaders(
    X_train,
    y_train,
    eval_set,
    weights,
    batch_size,
    num_workers,
    drop_last,
    pin_memory,
    need_shuffle=False
):
    """
    The dataloader class will sort the input classification result

    Parameters
    ----------
    X_train : 2D array
        The input matrix of classification result of shallow model and deep model (batch, 2 x classification dim)
    y_train : 2D array
        The correctness of classification of shallow model and deep model, and the groundtruth index of this classification (batch, 3)

    Return format in each iter
    ----------
    X: list of tensors [cls result of shallow model: (batch,classification dim), cls result of deep model: (batch,classification dim)]
    Y: tensor (batch, 3)

    """
    train_dataloader = DataLoader(
        TSP_Dataset(X_train.astype(np.float32), y_train),
        batch_size=batch_size,
        sampler=None,
        num_workers=num_workers,
        drop_last=drop_last,
        pin_memory=pin_memory,
        shuffle=need_shuffle
    )

    valid_dataloaders = []
    for X, y in eval_set:
        valid_dataloaders.append(
            DataLoader(
                TSP_Dataset(X.astype(np.float32), y),
                batch_size=batch_size,
                num_workers=num_workers,
                pin_memory=pin_memory,
                shuffle=need_shuffle
            )
        )

    return train_dataloader, valid_dataloaders


'''
Possible define position:
    metrics.py line 116 & abstract_model line 591
Possible call position:
    abstract_model.py line 537 (_predict_epoch)
'''


class TSP_Loss_C_Metric(Metric):

    def __init__(self):
        self._name = "TSP-Loss-C-Metric"
        # Higer is better or lower is better
        self._maximize = False
        self.loss_fn = TSP_Loss_C()

    def __call__(self, scores, y_true):

        # Deprecated
        '''
        score_shallow = scores[0]
        score_deep = scores[1]
        loss = np.sum(np.square(score_shallow-y_true[:, [0]]) + np.square(score_deep-y_true[:, [1]]))/score_shallow.shape[0]
        '''

        score_shallow = scores[0]
        score_deep = scores[1]
        X = scores[2]

        loss = self.loss_fn(output_S=torch.tensor(score_shallow, dtype=torch.float32),
                            output_D=torch.tensor(score_deep, dtype=torch.float32),
                            X=X,
                            targets=torch.tensor(y_true, dtype=torch.float32))

        loss = loss.cpu().item()

        return loss

class TSP_recorrect_rate_Metric(Metric):

    def __init__(self):
        self._name = "AIR"
        # Higer is better or lower is better
        self._maximize = True

    def __call__(self, scores, y_true):
        
        score_shallow = scores[0]
        score_deep = scores[1]
        X = scores[2]

        # the index of condition that one of shallow or deep cls result is incorrect
        target_index=(y_true[:,0]!=y_true[:,1])

        # target shallow score
        target_shallow_score = score_shallow[target_index]

        # target deep score
        target_deep_score = score_deep[target_index]
        
        # Only keep index of incorrect classification of either the shallow model or the deep model
        y_true=y_true[target_index]

        # The total number of incorrect classification of either the shallow model or the deep model
        total_count = len(y_true)

        # The number of estimator give larger socre to correct cls result
        estimator_correct_count=0

        for i in range(total_count):
            # Find which one is correct
            # shallow model is correct
            if y_true[i,0]==1:
                if target_shallow_score[i,0]>target_deep_score[i,0]:
                    estimator_correct_count+=1

            # deep model is correct
            else:
                if target_deep_score[i,0]>target_shallow_score[i,0]:
                    estimator_correct_count+=1

        return estimator_correct_count/total_count


class TSP_Loss_A_Metric(Metric):

    def __init__(self):
        self._name = "TSP-Loss-A-Metric"
        # Higer is better or lower is better
        self._maximize = False
        self.loss_fn = TSP_Loss_A()

    def __call__(self, scores, y_true):
        score_shallow = scores[0]
        score_deep = scores[1]
        X = scores[2]

        loss = self.loss_fn(output_S=torch.tensor(score_shallow, dtype=torch.float32),
                            output_D=torch.tensor(score_deep, dtype=torch.float32),
                            X=X,
                            targets=torch.tensor(y_true, dtype=torch.float32))

        loss = loss.cpu().item()

        return loss


class TSP_Loss_Metric(Metric):

    def __init__(self):
        self._name = "TSP-Loss-Overall"
        # Higer is better or lower is better
        self._maximize = False
        self.loss_A = TSP_Loss_A_Metric()
        self.loss_C = TSP_Loss_C_Metric()

    def __call__(self, scores, y_true):

        lossA = self.loss_A(scores, y_true)

        lossC = self.loss_C(scores, y_true)

        loss =  TSP_Loss.Loss_A_weight * lossA + TSP_Loss.Loss_C_weight * lossC

        return loss


'''
Possible define position:
    abstract_model.py line 185 
Possible call position:
    abstract_model.py line 498 (_train_batch)
'''


class TSP_Loss_C(nn.Module):
    def __init__(self):
        super(TSP_Loss_C, self).__init__()
        self.device = torch.device(define_device("auto"))

    def forward(self, output_S, output_D, X, targets):
        """
        Compute LOSS C of TSP

        Parameters
        ----------
        output_S : tensor
            shape:(batch, 1)
        output_D : tensor
            shape:(batch, 1)
        X : list of tensor
            shape:[(batch,cls dim),(batch,cls dim)]
        targets: tensor
            shape:(batch, 3)

        Shape refer to TSP_implement.train_batch
        """

        # make sure they are in same device
        output_S = output_S.to(self.device)
        output_D = output_D.to(self.device)
        targets = targets.to(self.device)

        loss = ((output_S - targets[:, 0].view(-1, 1)).square() + (output_D - targets[:, 1].view(-1, 1)).square()).sum() / output_S.shape[0]

        return loss


class TSP_Loss_A(nn.Module):
    def __init__(self):
        super(TSP_Loss_A, self).__init__()
        self.device = torch.device(define_device("auto"))

    def forward(self, output_S, output_D, X, targets):
        """
        Compute LOSS A of TSP

        Parameters
        ----------
        output_S : tensor
            shape:(batch, 1)
        output_D : tensor
            shape:(batch, 1)
        X : list of tensor
            shape:[(batch,cls dim),(batch,cls dim)]
        targets: tensor
            shape:(batch, 3)

        Shape refer to TSP_implement.train_batch
        """

        # make sure they are in same device
        output_S = output_S.to(self.device)
        output_D = output_D.to(self.device)
        targets = targets.to(self.device)

        # (batch, 1)
        shallow_model_cor_prob = X[0].to(self.device).gather(dim=1, index=targets[:, 2].long().view(-1, 1).to(self.device))
        deep_model_cor_prob = X[1].to(self.device).gather(dim=1, index=targets[:, 2].long().view(-1, 1).to(self.device))

        # if correct class prob of shallow result > correct class prob of deep result
        loss_A_upper_term = shallow_model_cor_prob-deep_model_cor_prob-output_S+output_D
        loss_A_upper_term_index = (shallow_model_cor_prob > deep_model_cor_prob)
        loss_A_upper_term = loss_A_upper_term*loss_A_upper_term_index

        # otherwise
        loss_A_down_term = deep_model_cor_prob-shallow_model_cor_prob-output_D+output_S
        loss_A_down_term_index = (deep_model_cor_prob >= shallow_model_cor_prob)
        loss_A_down_term = loss_A_down_term*loss_A_down_term_index

        # Loss term A
        loss = torch.maximum((loss_A_upper_term + loss_A_down_term), torch.tensor(0).to(self.device)).sum()/output_S.shape[0]

        return loss

# For abstract_model.py line 467


class TSP_Loss(nn.Module):

    Loss_A_weight = 0.2
    Loss_C_weight = 1.0

    def __init__(self,lambda_CDC_weight,Loss_Base_weight=1):
        super(TSP_Loss, self).__init__()
        TSP_Loss.Loss_A_weight = lambda_CDC_weight
        TSP_Loss.Loss_C_weight = Loss_Base_weight

    def forward(self, output_S, output_D, X, targets):
        """
        Compute general LOSS  of TSP

        Parameters
        ----------
        output_S : tensor
            shape:(batch, 1)
        output_D : tensor
            shape:(batch, 1)
        X : list of tensor
            shape:[(batch,cls dim),(batch,cls dim)]
        targets: tensor
            shape:(batch, 3)

        Shape refer to TSP_implement.train_batch
        """
        loss = TSP_Loss.Loss_A_weight * TSP_Loss_A()(output_S, output_D, X, targets) + TSP_Loss.Loss_C_weight * TSP_Loss_C()(output_S, output_D, X, targets)

        return loss


def train_batch(X,
                y,
                network,
                loss_fn,
                device,
                lambda_sparse,
                clip_value,
                optimizer):
    """
    Trains one batch of data

    Parameters
    ----------
    X : torch.Tensor
        From TSP dataloader, list of tensor, shape:[(batch,cls dim),(batch,cls dim)]
    y : torch.Tensor
        From TSP dataloader, tensor, shape:(batch,3)

    Returns
    -------
    batch_outs : dict
        Dictionnary with "y": target and "score": prediction scores.
    batch_logs : dict
        Dictionnary with "batch_size" and "loss".
    """
    X_S, X_D = X[0], X[1]
    batch_logs = {"batch_size": X_S.shape[0]}

    X_S = X_S.to(device).float()
    X_D = X_D.to(device).float()
    y = y.to(device).float()

    for param in network.parameters():
        param.grad = None

    output_S, M_loss_S = network(X_S)
    output_D, M_loss_D = network(X_D)

    loss = loss_fn(output_S, output_D, X, y)

    # Add the overall sparsity loss
    loss = loss - lambda_sparse * M_loss_S - lambda_sparse * M_loss_D

    # Perform backward pass and optimization
    loss.backward()
    if clip_value:
        clip_grad_norm_(network.parameters(), clip_value)
    optimizer.step()

    batch_logs["loss"] = loss.cpu().detach().numpy().item()

    return batch_logs


# For utils.py line 34 and abstract_model.py line 262


class PredictDataset(Dataset):
    pass


class TSP_TabNet(torch.nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        n_d=8,
        n_a=8,
        n_steps=3,
        gamma=1.3,
        cat_idxs=[],
        cat_dims=[],
        cat_emb_dim=1,
        n_independent=2,
        n_shared=2,
        epsilon=1e-15,
        virtual_batch_size=128,
        momentum=0.02,
        mask_type="sparsemax",
    ):
        """
        Defines TabNet network

        Parameters
        ----------
        input_dim : int
            Initial number of features
        output_dim : int
            Dimension of network output
            examples : one for regression, 2 for binary classification etc...
        n_d : int
            Dimension of the prediction  layer (usually between 4 and 64)
        n_a : int
            Dimension of the attention  layer (usually between 4 and 64)
        n_steps : int
            Number of successive steps in the network (usually between 3 and 10)
        gamma : float
            Float above 1, scaling factor for attention updates (usually between 1.0 to 2.0)
        cat_idxs : list of int
            Index of each categorical column in the dataset
        cat_dims : list of int
            Number of categories in each categorical column
        cat_emb_dim : int or list of int
            Size of the embedding of categorical features
            if int, all categorical features will have same embedding size
            if list of int, every corresponding feature will have specific size
        n_independent : int
            Number of independent GLU layer in each GLU block (default 2)
        n_shared : int
            Number of independent GLU layer in each GLU block (default 2)
        epsilon : float
            Avoid log(0), this should be kept very low
        virtual_batch_size : int
            Batch size for Ghost Batch Normalization
        momentum : float
            Float value between 0 and 1 which will be used for momentum in all batch norm
        mask_type : str
            Either "sparsemax" or "entmax" : this is the masking function to use
        """
        super(TSP_TabNet, self).__init__()
        print('TSP_TabNet!')
        self.cat_idxs = cat_idxs or []
        self.cat_dims = cat_dims or []
        self.cat_emb_dim = cat_emb_dim

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_d = n_d
        self.n_a = n_a
        self.n_steps = n_steps
        self.gamma = gamma
        self.epsilon = epsilon
        self.n_independent = n_independent
        self.n_shared = n_shared
        self.mask_type = mask_type

        if self.n_steps <= 0:
            raise ValueError("n_steps should be a positive integer.")
        if self.n_independent == 0 and self.n_shared == 0:
            raise ValueError("n_shared and n_independent can't be both zero.")

        self.virtual_batch_size = virtual_batch_size
        self.embedder = EmbeddingGenerator(input_dim, cat_dims, cat_idxs, cat_emb_dim)
        self.post_embed_dim = self.embedder.post_embed_dim
        self.tabnet = TabNetNoEmbeddings(
            self.post_embed_dim,
            output_dim,
            n_d,
            n_a,
            n_steps,
            gamma,
            n_independent,
            n_shared,
            epsilon,
            virtual_batch_size,
            momentum,
            mask_type,
        )

    def forward(self, x):
        # sort from largest to smmallest
        x, _ = torch.sort(x, dim=1, descending=True)

        x = self.embedder(x)
        return self.tabnet(x)

    def forward_masks(self, x):
        # sort from largest to smmallest
        x, _ = torch.sort(x, dim=1, descending=True)

        x = self.embedder(x)
        return self.tabnet.forward_masks(x)


if __name__ == '__main__':
    pass
