import argparse
import pickle
import os
from os.path import join
import numpy as np

def main():
    """
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir')
    parser.add_argument('--datadir')
    parser.add_argument('--dataset')

    parser.add_argument('--limit_train', default=None, type=int)
    parser.add_argument('--limit_test', default=None, type=int)

    args = parser.parse_args()

    dataset = args.dataset

    fds = os.path.join(args.datadir, args.dataset + ".pkl")
    ds = pickle.load(open(fds,'rb'))

    train_labels = ds['train_labels']
    test_labels  = ds['test_labels']

    if args.limit_train != None:
        train_labels = train_labels[:args.limit_train]

    if args.limit_test != None:
        test_labels = test_labels[:args.limit_test]
    
    n_test  = len(test_labels)
    n_train = len(train_labels)
    
    d = join(args.dir, args.dataset)

    assert(os.path.isdir(d))
    
    fset = set(os.listdir(d))
    N = len(fset)
    #must be {0,...,N-1}.pkl
    filenames = [str(i)+".pkl" for i in range(N)]
    assert(set(filenames) == fset)

    hyp = np.zeros_like(test_labels)
    top2_correct = np.zeros((n_test,),'bool')
    
    ctr = 0
    
    for f in filenames:
        full = join(d,f)
        v = pickle.load(open(full,'rb'))

        n_testi, tmp = v.shape
        if tmp != n_train:
            raise ValueError(f"bad shape:{v.shape} n_train:{n_train}")
        assert(tmp == n_train)

        args = np.argsort(v,axis=1)
        hypi = train_labels[args[:,0]]
        n_testi = len(hypi)

        idx_range = slice(ctr, ctr + n_testi)

        e0 = train_labels[args[:,0]] == test_labels[idx_range]
        e1 = train_labels[args[:,1]] == test_labels[idx_range]
        corr = np.logical_or(e0,e1)

        hyp[idx_range] = hypi
        top2_correct[idx_range] = corr
    
        ctr += n_testi

    acc = np.mean(hyp == test_labels)
    top2 = np.mean(top2_correct)
    res = {
        'name': dataset,
        'knn1': acc,
        'top2': top2,
    }
    print(res)

    
if __name__ == "__main__":
    main()
