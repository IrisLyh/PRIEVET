import argparse
import data
import metrics
import time
import triangleCount
 

def read_opts():
    '''
    read oprions
    1. dataset name --dataset
    
       file location:/PrivSubCnt/Data
       file name: name.txt
       data form: two rows with two end nodes
                      e.g.  id1 id2
                            id3 id4
                  undirected graphs
       dataset downloaded:
            -- wiki-Vote
            -- email-Enron
            -- cit-HepTh
            
       --All read all datasets into memory (all listed)
       --Default wiki-Vote


    2. user numbers --user_num

       the number of users reported their subgraphs

       --default 7115 [matched with wiki-Vote]

    3. perturb scheme --scheme

       the pertube method need to be used
            --CaliToUpper 
            --CaliToLS
            --CaliToTrunc
            --DDP
        --default --CaliToUpper
    '''
    parse = argparse.ArgumentParser()
    parse.add_argument('--dataset',
                        help='dataset name\n dataset downloaded:\n \t wiki-Vote\n \t email-Enron\n \tcit-HepTh\n \tfacebook',
                        type=str,
                        default='wiki-Vote')
  
    parse.add_argument('--scheme',
                        help='perturb scheme',
                        type=str,
                        default='CaliToUpper')

    parse.add_argument('--epsilon',
                        help='the overall privacy budget',
                        type=float,
                        default=1.0)

    parse.add_argument('--delta',
                        help='the overall failure rate',
                        type=float,
                        default=5e-6)

    parse.add_argument('--beta',
                        help='the privacy allocation for epsilon',
                        type=float,
                        default=0.2)

    parse.add_argument('--alpha',
                        help='parameter for privacy budget allocation of epsilon_1',
                        type=float,
                        default=0.5)
    
    parse.add_argument('--h_prime',
                        help='the parameter for finding the maximum degree and the maximum global data correlation',
                        type = int,
                        default=100)

    parse.add_argument('--r',
                        help='the parameter for computing the upper bound of local sensitivity of 2PhasesRLDP',
                        type=int,
                        default=5)

    parse.add_argument('--p',
                        help='the sample rate for triangle subsample in algorithm 2PhasesRLDP',
                        type=float,
                        default=0.01)

   
    try:
        parsed = vars(parse.parse_args())
    except IOError as msg: parse.error(str(msg))

    return parsed

            



if __name__ == "__main__":
    #read options 
    gp = read_opts()
    sparse_dataset = data.dataset(params = gp)
    sparse_data = sparse_dataset.load_sparse_data()
    sparse_mat = sparse_data[gp['dataset']]['mat']
    g = metrics.sparseGraphMetrics()
    g.show_info(sparse_mat,gp['dataset'])
    rldp = triangleCount.RLDP(sparse_mat,gp)
    
    start_time = time.time()
    if gp['scheme'] == 'CaliToUpper':
        rldp.caliToUpper()
    elif gp['scheme'] == 'CaliToLS':
        rldp.caliToLS()
    elif gp['scheme'] == 'CaliToTrunc':
        rldp.caliToTrunc()
    elif gp['scheme'] == 'DDP':
        rldp.DDP()
    else:
        print(" + The input perturbation scheme is not defined.")
    end_time = time.time()
    print("+ counting finished. counting time:{} minutes.".format((end_time-start_time)/60))






    

    


