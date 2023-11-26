import numpy as np
import os
import time

from numpy.core.fromnumeric import shape
from scipy.sparse import coo_matrix, csr_matrix, csc_matrix
'''
1. Read and process dataset
2. All data is in form of .txt
3. A node is represented by node ID
4. Graphs are undirected, a tuple stand for an edge
5. Example
    1 2
    3 4
    represent adjacency matrix
    |---|---|---|---|
    | 0 | 1 | 0 | 0 |
    |---|---|---|---|
    | 1 | 0 | 0 | 0 |
    |---|---|---|---|
    | 0 | 0 | 0 | 1 |
    |---|---|---|---|
    | 0 | 0 | 1 | 0 |
    |---|---|---|---|
'''
class dataset:
    def __init__(self, params) -> None:
        '''
        input params, which is the global 
        if params['dataset'] = 'All'
        params['user_num'] has to be reset
        '''
        self.dataset = {}   # set self.dataset as a dictionary:{'name':{'mat':[[],[],...],'user_num': int}

        if params['dataset'] == 'wiki-Vote':
            user_num = 7115
            self.dataset[params['dataset']] = {'user_num':user_num}
        elif params['dataset'] == 'cit-HepTh':
            user_num = 27770
            self.dataset[params['dataset']] = {'user_num':user_num}
        elif params['dataset'] == 'email-Enron':
            user_num = 36692
            self.dataset[params['dataset']] = {'user_num':user_num}
        elif params['dataset'] == 'facebook':
            user_num = 4039
            self.dataset[params['dataset']] = {'user_num':user_num}
        elif params['dataset'] == 'IMDB':
            user_num = 896308
            self.dataset[params['dataset']] = {'user_num':user_num}
        else:
            print("+ Undefinded dataset name.")



    def load_sparse_data(self) -> None:
        '''
        1. Load adjacency matrix from file into dictionary
        2. Dataset is in form of dictionry:
           dataset = {'name':{'user_num': int, 'mat': nparray(2d,np.int64)}}
        '''
        dataset_name_list = self.dataset.keys() # get all dataset names
        for name in dataset_name_list:
            local_addr = str(os.getcwd()) + '/dataset/' + name + '.txt' # get the absolute path of the dataset file
            print("+ start loading dataset {} from {}".format(name, local_addr))
            column = []
            row = []
            value = []
            start_loading = time.time()
            with open(local_addr,'r') as dataset_file:
                edges = dataset_file.readlines()    # read all edges
                count = 0
                for edge in edges:          # process every edge
                    count += 1
                    edge = edge.split(' ')  # split lines with whitespace
                    A = int(edge[0])        # get two end nodes' ID
                    B = int(edge[1])
                    if A!=B:
                        column.append(A)
                        row.append(B)
                        value.append(True)
                        column.append(B)
                        row.append(A)
                        value.append(True)
            column = np.array(column)
            row = np.array(row)
            value = np.array(value)
            adj_mat = csr_matrix((value,(column,row)),shape=(self.dataset[name]['user_num'],self.dataset[name]['user_num']),dtype=np.int32)
            end_loading = time.time()
            print("+ finsh loading. loading time:{}".format(end_loading-start_loading))
            self.dataset[name]['mat'] = adj_mat # update the gobal dictionary
        return self.dataset
        
