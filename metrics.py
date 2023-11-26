import numpy as np
# import tools
from scipy import sparse
import os

class sparseGraphMetrics(object):
    def __init__(self):
        pass


    def triSen(self,csr_matrix,dataset):
        '''
        Local sensitivity of triangle counting.
        '''
        path = './dataset/statistics/senCnt/'
        file_name = path + dataset + '.npz'
        if os.path.exists(file_name):
            sparse_local_sensitivity = sparse.load_npz(file_name)
            local_sensitivity = sparse_local_sensitivity.toarray()
        else:
            if not os.path.exists(path):
                os.makedirs(path)
            local_sensitivity = []
            for i in range(csr_matrix.shape[0]):
                neighbor_vector = csr_matrix[i]  # get the i-th row
                sen_per_edge_vector = neighbor_vector @ csr_matrix
                sen_per_edge_vector.setdiag(0,i)
                tmp = sen_per_edge_vector.max(axis=None, out=None)
                local_sensitivity.append(tmp)
                if i%10000==0:
                    print("process:{}".format(i))
            local_sensitivity = np.array(local_sensitivity)
            sparse_local_sensitivity = sparse.csr_matrix(local_sensitivity)
            sparse.save_npz(file_name,sparse_local_sensitivity)
        return local_sensitivity


    def triCnt(self,csr_matrix,dataset):
        '''
        Counting triangles based on crs_matrix.
        '''
        path = './dataset/statistics/triCnt/'
        file_name = path + dataset + '.npz'
        if os.path.exists(file_name):
            sparse_local_triangle_count = sparse.load_npz(file_name)
            local_triangle_count = sparse_local_triangle_count.toarray()
        else:
            if not os.path.exists(path):
                os.makedirs(path)
            local_triangle_count = []
            for i in range(csr_matrix.shape[0]):
                neighbor_vector = csr_matrix[i] # get the i-th row
                sen_per_edge_vector = neighbor_vector @ csr_matrix
                tmp = sen_per_edge_vector.multiply(neighbor_vector).sum()/2
                local_triangle_count.append(tmp)
                if i%10000==0:
                    print("process:{}".format(i))
            local_triangle_count = np.array(local_triangle_count)
            sparse_local_triangle_count = sparse.csr_matrix(local_triangle_count)
            sparse.save_npz(file_name,sparse_local_triangle_count)
            print(np.sum(local_triangle_count)/3)
        return local_triangle_count
    


    def nodDeg(self,csr_matrix):
        '''
        Node degree for each node

            1. sum on each row
            2. return the dense numpy matrix
        '''
        nodeDeg = csr_matrix.sum(axis=0)
        return np.array(nodeDeg).reshape(nodeDeg.shape[1])

    


            
    def smpTriCnt(self,csr_matrix,p,dataset):
        '''
        Computing the triangles after subsampling
            1. take sparse matrix
            2. return sparse matrix
        '''
        path = './dataset/statistics/smpTri/'
        file_name = path + dataset + '_' + str(p) + '.npz'

        if os.path.exists(file_name):
            smp_triangle_count = sparse.load_npz(file_name)

        else:
            if not os.path.exists(path):
                os.makedirs(path)
            def subsample(data,indices,p):
                '''indices is a segment of the whole indices sequences in accordance with data
                '''
                smp_data = []
                smp_indices = []
                for i in range(len(data)):
                    if data[i] > 0:
                        indicator = np.random.random(data[i])
                        cnt = len(np.where(indicator<=p/2)[0])
                        if cnt > 0:
                            smp_data.append(cnt)
                            smp_indices.append(indices[i])
                return smp_data,smp_indices

            value = []
            indices = []
            indptr = [0]
            tri_cnt = 0
            print("+ start subsampling")
            for i in range(csr_matrix.shape[0]):
                neighbor_vector = csr_matrix[i] # get the i-th row
                sen_per_edge_vector = neighbor_vector @ csr_matrix
                tmp = sen_per_edge_vector.multiply(neighbor_vector)
                tri_cnt += tmp.sum()
                data_i, indices_i = subsample(tmp.data,tmp.indices,p)
                value.extend(data_i)
                indices.extend(indices_i)
                indptr.append(len(value))
                if i%10000==0:
                        print("process:{}".format(i))

            
            smp_triangle_count = sparse.csr_matrix((value,indices,indptr),shape=csr_matrix.shape)
            print("+ finish subsampling. Subsampling error:{}".format(smp_triangle_count.sum()/p/(tri_cnt/2)))
            sparse.save_npz(file_name,smp_triangle_count)
        return smp_triangle_count


    def smpLocSen(self,csr_matrix,p,dataset):
        '''
        Computing the local sensitivity after subsampling
            1. take sparse matrix
            2. return sparse matrix
        '''
        path = './dataset/statistics/smpSen/'
        file_name = path + dataset + '_' + str(p) + '.npz'
        p = p/2
        if os.path.exists(file_name):
            smp_local_sensitivity = sparse.load_npz(file_name)

        else:
            if not os.path.exists(path):
                os.makedirs(path)
            def subsample(data,indices,p):
                '''indices is a segment of the whole indices sequences in accordance with data
                '''
                smp_data = []
                smp_indices = []
                for i in range(len(data)):
                    if data[i] > 0:
                        indicator = np.random.random(data[i])
                        cnt = len(np.where(indicator<=p)[0])
                        if cnt > 0:
                            smp_data.append(cnt)
                            smp_indices.append(indices[i])
                return smp_data,smp_indices

            value = []
            indices = []
            indptr = [0]
            avg_sen = 0
            print("+ start subsampling",)
            for i in range(csr_matrix.shape[0]):
                neighbor_vector = csr_matrix[i] # get the i-th row
                sen_per_edge_vector = neighbor_vector @ csr_matrix
                avg_sen += sen_per_edge_vector.sum()
                data_i, indices_i = subsample(sen_per_edge_vector.data,sen_per_edge_vector.indices,p)
                value.extend(data_i)
                indices.extend(indices_i)
                indptr.append(len(value))
                if i%10000==0:
                    print("process:{}".format(i))
            avg_sen = avg_sen/(csr_matrix.shape[0]**2)
            smp_local_sensitivity = sparse.csr_matrix((value,indices,indptr),shape=csr_matrix.shape)
            smp_avg_sen = smp_local_sensitivity.sum()/p/(csr_matrix.shape[0]**2)
            print("+ finish subsampling. Subsampling error:{}".format(np.abs(smp_avg_sen-avg_sen)/avg_sen))
            sparse.save_npz(file_name,smp_local_sensitivity)
        return smp_local_sensitivity

    

    def show_info(self,csr_matrix,dataset):
        num_of_nodes = csr_matrix.shape[0]
        node_deg = self.nodDeg(csr_matrix)
        print(node_deg.shape)
        num_of_edges = np.sum(node_deg)/2
        max_deg = np.max(node_deg)
        avg_deg = np.mean(node_deg)
        med_deg = np.median(node_deg)
        tri_cnt = np.sum(self.triCnt(csr_matrix,dataset))/3
        print("#Node:{}\tEdges:{}\t#Trianlges:{}\tMax d:{}\t Avg. d:{}\tMed. d:{}".format(num_of_nodes,num_of_edges,tri_cnt,max_deg,avg_deg,med_deg))