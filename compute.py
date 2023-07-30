"""
"""
import argparse
import pickle
import os
from os.path import join
import gzip
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import time
import json
from gziplength import GzipLengthCalc
from tqdm import tqdm

clen = lambda data : len(gzip.compress(data))
NCD = lambda c1, c2, c12 : (c12-min(c1,c2))/max(c1, c2)

def do_block(test_data, train_data,
             precomputed_lengths, outfile,
             method,
             passthrough):
    """
    train_lengths: pre-computed

    passthough dictionary is just .update() to the output.
    """
    start = time.time()
    n_test  = len(test_data)
    n_train = len(train_data)
    
    D = np.zeros((n_test,n_train))

    assert(method in (
        'orig',
        'precomputed',
        'gziplength',
        'zeros',
    ))

    #orig
    if method == 'orig':
        for i,t1 in enumerate(test_data):
            l1 = clen(t1.encode('utf8'))
            
            for j, t2 in enumerate(train_data):
                l2 = clen(t2.encode('utf8'))
                l12 = clen( (t1 + ' ' + t2).encode('utf8') )
                D[i,j] = NCD(l1, l2, l12)
    
    elif method == 'precomputed':
        for i,t1 in enumerate(test_data):
            l1 = clen(t1)
            
            for j, t2 in enumerate(train_data):
                l2 = precomputed_lengths[j]
                l12 = clen(t1 + b" " + t2)
                D[i,j] = NCD(l1, l2, l12)

    elif method == 'gziplength':
        for i,t1 in enumerate(test_data):
            g = GzipLengthCalc(t1)
            l1 = g.length1
            
            for j, t2 in enumerate(train_data):
                l2 = precomputed_lengths[j]
                l12 = g.length2(t2)
                D[i,j] = NCD(l1, l2, l12)
                
    elif method == 'zeros':
        #D is already zeros
        pass
    
    else:
        raise ValueError('bad method:' + repr(method))

    with open(outfile,'wb') as f:
        pickle.dump(D,f)
        
    out = {
        "outfile": outfile,
        "size"   : D.shape,
        "size_mb": os.stat(outfile).st_size/(2**20),
        "time"   : time.time() - start,
    }
    out.update(passthrough)
    return out

def done_callback(future):
    """
    """
    print("[done]", future.result())
    ex = future.exception()
    if ex:
        print("ERROR:",ex)
        
    
def main():
    """
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset',required=True,help="name of dataset")
    parser.add_argument('--datadir',required=True,help="load data from $datadir/$dataset.pkl")

    parser.add_argument('--outdir',required=True,help="write to $outdir/$dataset/")
    
    parser.add_argument('--method',
                        default='gziplength',
                        choices = [
                            'orig',
                            'precomputed',
                            'gziplength',
                            'zeros',
                        ])
    parser.add_argument('--splitsize', default=500, type=int)

    parser.add_argument('--limit_train', default=None, type=int)
    parser.add_argument('--limit_test', default=None, type=int)
    
    args = parser.parse_args()

    method = args.method
    
    
    outdir_ds = join(args.outdir,
                     args.dataset)
    
    if not os.path.isdir(outdir_ds):
        os.makedirs(outdir_ds)

    ds = pickle.load(open(os.path.join(
        args.datadir,
        args.dataset + ".pkl"),'rb'))
    print(ds.keys())

    if method in ('orig','zeros'):
        pass #keep as strings
        train_data = ds['train_data']
        test_data  = ds['test_data']
    else:
        # convert strings to bytes
        train_data = [t.encode('utf8') for t in ds['train_data']]
        test_data  = [t.encode('utf8') for t in ds['test_data']]

    if args.limit_train != None:
        train_data = train_data[:args.limit_train]

    if args.limit_test != None:
        test_data = test_data[:args.limit_test]

    n_train = len(train_data)
    n_test  = len(test_data)

    #pre-process train_data
    if method in ('orig','zeros'):
        train_lengths = None # not used
    else:
        train_lengths = []
        for j,t2 in enumerate(tqdm(train_data)):
            train_lengths.append(clen(t2))

    splitsize = args.splitsize
    start_indices = list(range(0, n_test, splitsize))

    num_splits = len(start_indices)

    ith_file = lambda i : os.path.join(outdir_ds, str(i) + ".pkl")
    
    #or multiprocessing.cpu_count()
    ncpu = os.cpu_count()

    max_workers = ncpu

    print(json.dumps(dict(
        n_train = n_train,
        n_test = n_test,
        num_splits = num_splits,
        ncpu = ncpu,
        max_workers = max_workers,
        method = method,
        splitsize = splitsize,
    )))

    with ProcessPoolExecutor(max_workers=max_workers) as executor:

        for i,k in enumerate(start_indices):
            
            future = executor.submit(
                do_block,
                test_data[k:k+splitsize],
                train_data,
                train_lengths,
                ith_file(i),
                method,
                {"num_splits":num_splits},
            )
            future.add_done_callback(done_callback)
        
if __name__ == "__main__":
    main()
