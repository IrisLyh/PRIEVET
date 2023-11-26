from cmath import sqrt
from math import floor
import numpy as np
import metrics
# import tools
import math
import numpy as np
from scipy import sparse

class RLDP(object):
    def __init__(self,data,params) -> None:
        self.data = data                # this should be a csr_matrix
        self.dname = params['dataset']
        self.scheme = params['scheme']
        self.epsilon = params['epsilon']
        self.delta = params['delta']
        self.beta = params['beta']
        self.h_prime = params['h_prime']
        self.alpha = params['alpha']
        self.r = params['r']
        self.p = params['p']
        # the number of clients
        self.n = self.data.shape[0]
        self.graph = metrics.sparseGraphMetrics()
    
    def lapUpperBound(self,x,lapScale,delta):
        return x+lapScale*np.log(1.0/(2*delta))
    
    def correlationEstimation(self,eps,delta):
        nodeDeg = self.graph.nodDeg(self.data)
        deg_prime = nodeDeg + np.random.laplace(loc=0,scale=2/(eps*self.alpha),size=nodeDeg.shape)

        # get the upper bound of noisy node degree with lemma 1
        scale_deg = 2/(self.alpha*eps)
        deg_upper = self.lapUpperBound(deg_prime,scale_deg,delta/(self.h_prime+1))
        # print("deg_upper.shape:{}".format(deg_upper.shape))
        senCnt = self.graph.triSen(self.data,self.dname).T
        sort_index = np.argsort(-deg_upper)
        index = 0
        for i in range(0,self.h_prime):
            j=i+1
            if j/(self.alpha*eps)*np.log((self.h_prime+1)/delta) >= deg_upper[sort_index[j+2]]:
                index = floor(j/2.0)
                break

        sen_prime = senCnt + np.random.laplace(loc=0,scale=index,size=senCnt.shape)
        # get the upper bound of the noisy sensitivity with lemma 1
        scale_sen = index/((1-self.alpha)*eps)
        sen_upper = self.lapUpperBound(sen_prime,scale_sen,delta/(self.h_prime+1))
        # print("sen_upper:{}".format(np.sen_upper))

        k = np.zeros(shape = (index+1,),dtype=int)
        for j in range(0,index+1):
            a = deg_upper[sort_index[j]]
            b = sen_upper[sort_index[j]]
            k[j] = min(a,b)
            # print("in correlation estimation. index:{}\ta:{}\tb:{}".format(j,a,b))
        maxSen = np.max(k)
        return maxSen

    

    def DDP(self):
        '''
        1. find the global maximum sensitivity
        2. perturb each local triangle count
        '''
        print("+ excuting CaliToUpper...")
        trialNum=300
        triCnt = self.graph.triCnt(self.data,self.dname)
        k=self.correlationEstimation(self.beta*self.epsilon,self.delta)
        print(" ++ global data correlation (k):{},\ttotal privacy budget epsilon:{}\tfialure rate delta:{},\teps_2^prime:{}".format(k,self.epsilon,self.delta,(1-self.beta)*self.epsilon))
        Delta_f = 3*k
        MRE = 0
        var = 0
        for i in range(0,trialNum):
            # print("privacy scale:{}".format(3*k/(1-self.beta)/self.epsilon))
            data_prime = triCnt + np.random.laplace(loc=0,scale=Delta_f/((1-self.beta)*self.epsilon),size=triCnt.shape)
            MRE = MRE + np.abs(np.sum(data_prime)-np.sum(triCnt))/np.sum(triCnt)
            var = var + (np.sum(data_prime)/3-np.sum(triCnt)/3)**2

        MRE = MRE/trialNum
        var = var/trialNum
        print("++ variance for DDP:{}".format(var))
        print("++ MRE for DDP:{}".format(MRE))
        return 0



    def privacyAllocation(self,eps,k,p):
        '''
        Compute eps_2^prime with dichotomy
        '''
        eps_low = 0
        eps_high = 200
        eps_middle = (eps_low + eps_high)/2
        eps_temp = 2*eps_middle+np.log(1+p*(np.exp(eps_middle)-1))*sqrt(2*(k-2)*np.log(1/self.delta))+(k-2)*np.log(1+p*(np.exp(eps_middle)-1))*p*(np.exp(eps_middle)-1)/(p*(np.exp(eps_middle)+1)+2)
        while np.abs(eps_temp - eps)>=1e-5:
            if eps_temp > eps:
                eps_high = eps_middle
            elif eps_temp < eps:
                eps_low = eps_middle
            eps_middle = (eps_low + eps_high)/2
            eps_temp = 2*eps_middle+np.log(1+p*(np.exp(eps_middle)-1))*sqrt(2*(k-2)*np.log(1/self.delta))+(k-2)*np.log(1+p*(np.exp(eps_middle)-1))*p*(np.exp(eps_middle)-1)/(p*(np.exp(eps_middle)+1)+2)
            # print(eps_temp)
        return eps_middle
    


    def caliToUpper(self):
        '''
        1. collect global correlation
        2. subsample
        3. collection triangle count
        '''
        print("+ excuting CaliToUpper...")
        eps1 = self.beta*self.epsilon
        eps2 = (1-self.beta)*self.epsilon
        k = self.correlationEstimation(eps1,self.delta) + 2
        k = math.ceil(k*self.p+self.r*(np.sqrt(k*self.p*(1-self.p)).real))
        #eps_prime that considers both sampling and compostion
        eps_prime = self.privacyAllocation(eps2,k,self.p)
        print(" ++ global data correlation (k):{},\ttotal privacy budget epsilon:{}\tfialure rate delta:{},\teps_2^prime:{}".format(k,self.epsilon,self.delta,eps_prime))

        # actual local sensitivity
        # get triangle count pre edge
        # average local sensitivity after pertuabtion subsample
        triCnt = self.graph.triCnt(self.data,self.dname)
        deg = self.graph.nodDeg(self.data)
        deg_prime = deg + np.random.laplace(loc=0,scale=2/(self.alpha*eps1),size=deg.shape)
        
        # get the upper bound of the node degree
        scale_deg = 1/(eps1*self.alpha/2)
        deg_upper = self.lapUpperBound(deg_prime,scale_deg,self.delta/(self.h_prime+1))

        # get the upper bound of local sensitivity after subsampling
        smp_loc_sen = deg_upper*self.p
        smp_loc_sen_upper = smp_loc_sen + (self.r*sqrt(k*self.p*(1-self.p))).real


        print(" ++ average local sensitivity after subsampling and perturbation for {} is:{}.\t(with ptrivacy budget for local sensitivity perturbation equals to {})".format(self.dname, np.mean(smp_loc_sen_upper),eps1/2))
        
            
        # get the raw local triangle count after subsampling
        smp_tri_cnt_per_edge = self.graph.smpTriCnt(self.data,self.p,self.dname)
        sample_error = np.abs(np.sum(triCnt) - np.sum(smp_tri_cnt_per_edge)/self.p)/np.sum(triCnt)
        print(" ++ subsample error:{}".format(sample_error))
    

        trialNum = 300
        MRE = 0
        var = 0
        # local triangle count for each node after subsampling
        smp_tri_cnt = smp_tri_cnt_per_edge.sum(axis = 1).reshape((smp_tri_cnt_per_edge.shape[0],))
 
        for i in range(0,trialNum):
            tri_cnt_prime = smp_tri_cnt + np.random.laplace(loc=0,scale = smp_loc_sen_upper/eps_prime,size=smp_tri_cnt.shape)
            MRE = MRE + np.abs(np.sum(tri_cnt_prime)/self.p-np.sum(triCnt))/np.sum(triCnt)
            var = var+(np.sum(tri_cnt_prime)/self.p/3-np.sum(triCnt)/6)**2
        MRE = MRE/trialNum
        var = var/trialNum
        print(" ++ variance for CaliToUpper:{}".format(var))
        print(" ++ MRE for CaliToUpper:{}".format(MRE))
        return 0

    
    def caliToLS(self):
        '''
        1. return the result of perturbation with the true sensitivity after perturbation
        2. put trials into this function to save time
        '''

        print("+ excuting CaliToLS...")
        
        # prepare the privacy budget
        eps1 = self.epsilon*self.beta
        eps2 = self.epsilon*(1-self.beta)

        k = self.correlationEstimation(eps1,self.delta) + 2
        print("test k:{}".format(k))
        k = math.ceil(k*self.p+self.r*(np.sqrt(k*self.p*(1-self.p)).real))

        #privacy budget allocation
        eps_prime = self.privacyAllocation(eps2,k-2,self.p)
        if eps_prime > np.log(2):
            eps_prime -= np.log(2)
        else:
            eps_prime = 0
        print(" ++ global data correlation (k):{},\ttotal privacy budget epsilon:{}\tfialure rate delta:{},\teps_2^prime:{}".format(k,self.epsilon,self.delta,eps_prime))

        
        # prepare the data
        triCnt = self.graph.triCnt(self.data,self.dname)
        smp_loc_sen_per_edge = self.graph.smpLocSen(self.data,self.p,self.dname)
        smp_tri_cnt_per_edge = self.graph.smpTriCnt(self.data,self.p,self.dname)


        smp_loc_sen = smp_loc_sen_per_edge.max(axis=1).todense()
        smp_tri_cnt = np.array(smp_tri_cnt_per_edge.sum(axis=1))

        sample_error = np.abs(np.sum(smp_tri_cnt)/self.p-np.sum(triCnt))/np.sum(triCnt)
        print(" ++ subsample error:{}".format(sample_error))

        
        # set the minimum sensitivity to 1
        mask = smp_loc_sen == 0
        # print(smp_loc_sen.shape)
        smp_loc_sen[mask] = 1
        print(" ++ maximum average local sensitivity after subsampling for {} is {}.".format(self.dname, np.max(smp_loc_sen)))
        print(" ++ minimum local sensitivity after subsampling for {} is: {}.".format(self.dname, np.min(smp_loc_sen)))
        print(" ++ median of local sensitivity after subsampling for {} is: {}".format(self.dname, np.median(np.array(smp_loc_sen))))
        print(" ++ average local sensitivity after subsampling for {} is: {}.".format(self.dname, np.mean(smp_loc_sen)))

        
        trialNum = 300
        MRE = 0
        var = 0
        if eps_prime > 0:
            for i in range(0, trialNum):
                data_prime = smp_tri_cnt + np.random.laplace(loc=0, scale = smp_loc_sen/eps_prime,size=smp_tri_cnt.shape)
                MRE = MRE + np.abs(np.sum(data_prime)/self.p-np.sum(triCnt))/np.sum(triCnt)
                var = var + (np.sum(data_prime)/self.p/3-np.sum(triCnt)/3)**2
            var = var/trialNum
            MRE = MRE/trialNum
            print(" ++ variance for CaliToLS:{}".format(var))
            print(" ++ MRE for CaliToLS:{}".format(MRE))
        return 0


    def caliToTrunc(self):
        '''
        1. truncated the tirangle on each edge
        2. truncation boundary: node degree + laplace nosie + constant
        3. check for each edge: whether the trangle count for each edge is above the boundary or not
        '''
        
        print("+ excuting CaliToTrunc...")

        # prepare the privacy budget
        
        delta = self.delta*5
        eps1 = self.epsilon*self.beta
        eps2 = self.epsilon*(1-self.beta)

        # privacy budget allocation
        k = self.correlationEstimation(eps1,delta) + 2
        k = math.ceil(k*self.p+self.r*(np.sqrt(k*self.p*(1-self.p)).real))
        print(k)
       
        eps_prime = self.privacyAllocation(eps2,k,self.p)
        print(" ++ global data correlation (k):{},\ttotal privacy budget epsilon:{}\tfialure rate delta:{},\teps_2^prime:{}".format(k,self.epsilon,self.delta,eps_prime))

        
        
        deg = self.graph.nodDeg(self.data)
        deg_prime = deg + np.random.laplace(loc=0,scale=2/(eps1*self.alpha),size=deg.shape)
        triCnt = self.graph.triCnt(self.data,self.dname)
        smp_tri_cnt_per_edge = self.graph.smpTriCnt(self.data,self.p,self.dname)

        sample_error = np.abs(smp_tri_cnt_per_edge.sum()/self.p-np.sum(triCnt))/np.sum(triCnt)
        print(" ++ subsample error:{}".format(sample_error))


        
        
        # threshold
        if self.dname == 'facebook' or self.dname == 'IMDB':
            T = np.maximum(np.floor((deg_prime + 2/eps1/self.alpha*np.log(1/2/self.delta))*self.p/0.75),1)    # a larger threshold
        else:
            T = np.maximum(np.floor((deg_prime + 2/eps1/self.alpha*np.log(1/2/self.delta))*self.p/2),1)
        
        threshold_value = []
        for i in range(0,self.n):
            threshold_value.extend([T[i]]*(smp_tri_cnt_per_edge.indptr[i+1]-smp_tri_cnt_per_edge.indptr[i]))
        threshold_value = np.array(threshold_value)
        threshold = sparse.csr_matrix((threshold_value,smp_tri_cnt_per_edge.indices,smp_tri_cnt_per_edge.indptr),shape=smp_tri_cnt_per_edge.shape)

        print(" ++ the average threshold:{}".format(np.sum(T)/T.shape[0]))

        
        trialNum = 300
        MRE = 0
        var = 0
        MRE_without_cali = 0
        truncation_numbers = 0
        truncation_bias = 0
        eps_prime /= 2

        for i in range(0, trialNum):
            # split users into two halves for triangle count report and truncation bias report, respectively
            if i%50 == 0:
                print("     +++ at round:{}".format(i))
            truncated_cnt_per_edge = np.minimum(smp_tri_cnt_per_edge.data,threshold.data)

            # triangle count per node
            idx = np.where(smp_tri_cnt_per_edge.indptr==smp_tri_cnt_per_edge.indptr[-1])[0][0]
            truncated_cnt = np.zeros(shape=(smp_tri_cnt_per_edge.shape[0],))
            truncated_cnt[:idx] = np.add.reduceat(truncated_cnt_per_edge,smp_tri_cnt_per_edge.indptr[:idx])
            truncated_cnt[np.diff(smp_tri_cnt_per_edge.indptr)==0]=0
 


            # perturb truncated triangle count
            truncated_cnt_prime = truncated_cnt + np.random.laplace(loc=0,scale=T/eps_prime,size=truncated_cnt.shape)
            
            # thrown-away triangles
            thrown_away_per_edge = np.minimum(np.maximum(smp_tri_cnt_per_edge.data-threshold.data,0),threshold.data)
            
            # thrown-away triangles per node
            thrown_away_cnt = np.zeros(shape=(smp_tri_cnt_per_edge.shape[0],))
            thrown_away_cnt[:idx] = np.add.reduceat(thrown_away_per_edge,smp_tri_cnt_per_edge.indptr[:idx])
            thrown_away_cnt[np.diff(smp_tri_cnt_per_edge.indptr==0)]=0

            truncation_bias += np.abs((np.sum(truncated_cnt))/self.p-np.sum(triCnt))/np.sum(triCnt)

            # perturb thrown away triangle count
            noise = np.random.laplace(loc=0,scale=T/eps_prime,size=thrown_away_cnt.shape)
            thrown_away_cnt_prime = thrown_away_cnt + noise


            truncation_numbers += np.sum(thrown_away_cnt/self.p/3)
            


            MRE_without_cali += np.abs(np.sum(truncated_cnt_prime)/self.p-np.sum(triCnt))/np.sum(triCnt)
            MRE += np.abs((np.sum(truncated_cnt_prime)+np.sum(thrown_away_cnt_prime))/self.p-np.sum(triCnt))/np.sum(triCnt)
            var += ((np.sum(truncated_cnt_prime)+np.sum(thrown_away_cnt_prime))/self.p/3-np.sum(triCnt)/3)**2
        
        truncation_bias /= trialNum
        MRE_without_cali /=trialNum
        MRE /= trialNum
        var /= trialNum
        truncation_numbers /= trialNum
            
        print(" ++ thrown-away triangles for CaliToTrunc:{}".format(truncation_numbers))
        print(" ++ truncation bias for CaliToTrunc:{}".format(truncation_bias))
        print(" ++ variance for CaliToTrunc:{}".format(var))
        print(" ++ MRE without calibration for CaliToTrunc:{}".format(MRE_without_cali))
        print(" ++ MRE with calibration for CaliToTrunc:{}".format(MRE))
        print("row count:{}".format(np.sum(triCnt)/3))

        return 0