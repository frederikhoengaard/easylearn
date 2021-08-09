import numpy as np
import torch

from sklearn.model_selection import train_test_split


def dataset_split(
        features,
        labels,
        test_size=0.2,
        shuffle=True,
        random_state=None,
        stratify=False,
        to_tensor=False
    ) -> list:
    X_train, X_test, y_train, y_test = train_test_split(features,labels,test_size=test_size,shuffle=shuffle,random_state=random_state,stratify=stratify)
    if to_tensor:
        return torch.FloatTensor(X_train), torch.FloatTensor(X_test), torch.LongTensor(y_train), torch.LongTensor(y_test)
    else:
        return X_train, X_test, y_train, y_test



def main():
    pass


if __name__ == '__main__':
    main()