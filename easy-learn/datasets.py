import numpy as np
import pandas as pd
from sys import argv
from utils import Bunch

def _load_iris(return_X_y: bool = False, as_frame: bool = False):
    # TODO: We want this to return numerical targets

    bunch = Bunch()
   # bunch['target_names']
    frame_data = pd.read_csv('data/iris.csv',delimiter=',')
    non_frame_data = np.loadtxt('data/iris.csv',delimiter=',',skiprows=1)
    if as_frame:
        return frame_data
    else:
        bunch['data'] = non_frame_data[:,:-1]
        return bunch.data








def main():
    if 'ut' in argv:
        print(_load_iris(as_frame=True))

    
    from sklearn.datasets import load_iris

    data = load_iris()
    print(data.target_names)
    print(data.target)



if __name__ == '__main__':
    main()