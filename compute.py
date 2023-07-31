"""
"""
import os
from os.path import join
import argparse
import pickle
import gzip
import time
import json
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from tqdm import tqdm

from gziplength import GzipLengthCalc


clen = lambda data : len(gzip.compress(data))
NCD = lambda c1, c2, c12 : (c12-min(c1,c2))/max(c1, c2)

def do_block(test_data, train_data,
             precomputed_lengths,
             #outfile,
             method,
             i,
             num_save,
             args_dtype,
             passthrough):
    """
    train_lengths: pre-computed

    passthough dictionary is just .update() to the output.
    """
    start = time.time()
    n_test  = len(test_data)
    n_train = len(train_data)

    print(f"[start] {i=} {n_test=} {n_train=}")
    sys.stdout.flush()
    
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

    out = {
        "size"   : D.shape,
        "time"   : time.time() - start,
        "i": i,
    }
    out.update(passthrough)
    print("out:", out)

    top_args = np.argsort(D,axis=1)[:,:num_save].astype(args_dtype)

    return top_args

def done_callback(future):
    """
    """
    print("[done]")
    sys.stdout.flush()
    ex = future.exception()
    if ex:
        print("ERROR:",ex)
        
    
def main():
    """
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset',
                        required=True,help="path of .pkl of dataset")

    parser.add_argument('--method',
                        default='gziplength',
                        choices = [
                            'orig',
                            'precomputed',
                            'gziplength',
                            'zeros',
                        ])
    parser.add_argument('--splitsize', default=500, type=int)

    parser.add_argument('--num_save', default=100, type=int,
                        help = "number of nearest neighbors to save")
    
    parser.add_argument('--limit_train', default=None, type=int)
    parser.add_argument('--limit_test', default=None, type=int)

    parser.add_argument('--outfile',
                        help = "output path for .pkl of sorted indices")
    
    args = parser.parse_args()

    method = args.method
    
    ds = pickle.load(open(args.dataset,"rb"))
    print(ds.keys())

    if method in ('orig','zeros'):
        pass #keep as strings
        train_data = ds['train_data']
        test_data  = ds['test_data']
    else:
        # convert strings to bytes
        train_data = [t.encode('utf8') for t in ds['train_data']]
        test_data  = [t.encode('utf8') for t in ds['test_data']]

    train_labels = ds['train_labels']
    test_labels  = ds['test_labels']
    
    if args.limit_train != None:
        train_data   = train_data[:args.limit_train]
        train_labels = train_labels[:args.limit_train]

    if args.limit_test != None:
        test_data = test_data[:args.limit_test]
        test_labels = test_labels[:args.limit_test]

    n_train = len(train_data)
    n_test  = len(test_data)

    #pre-process train_data
    if method in ('orig','zeros'):
        train_lengths = None # not used
    else:
        train_lengths = []
        for j,t2 in enumerate(tqdm(train_data)):
            train_lengths.append(clen(t2))

    num_save = args.num_save
    splitsize = args.splitsize
    start_indices = list(range(0, n_test, splitsize))

    num_splits = len(start_indices)

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

    #
    args_dtype = 'uint32'

    futures = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:

        for i,k in enumerate(start_indices):

            future = executor.submit(
                do_block,
                test_data[k:k+splitsize],
                train_data,
                train_lengths,
                method,
                i,
                num_save,
                args_dtype,
                {"num_splits":num_splits},
            )
            future.add_done_callback(done_callback)
            futures.append((k,future))

    print("#futures:", len(futures))

    top_args = np.zeros((n_test,num_save), args_dtype)

    for k,future in futures:
        top_args1 = future.result()

        n_test1 = top_args1.shape[0]
        top_args[k:k+n_test1] = top_args1

    #compute 1st nearest neighbor score:
    hyp = train_labels[top_args[:,0]]
    ref = test_labels
    acc = (hyp == ref).mean()
    print(f"acc:{acc:0.3f}")

    pickle.dump({
        'train_labels': train_labels,
        'test_labels':  test_labels,
        'args': top_args,
    }, open(args.outfile,'wb'))
    print("wrote",args.outfile)
        
if __name__ == "__main__":
    main()
