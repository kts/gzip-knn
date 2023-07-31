"""
This needs to import "data" from https://github.com/bazingagin/npc_gzip

Save all datasets as pickle files to $outdir/$name.pkl

 data['train_data'] : [str]
 data['test_data'] : [str]
 data['train_labels'] : ndarray (n_train,) dtype=uint32
 data['test_labels']  : ndarray (n_test,)  dtype=uint32

"""
from data import (
    load_kinnews,
    load_kirnews,
    load_filipino,
    load_swahili,
    load_20news,
    )
import torchtext.datasets
import os
import pickle
import numpy as np
import argparse

#needed to add for SogouNews dataset:
import sys
import csv
print("SET csv.field_size_limit:", sys.maxsize)
csv.field_size_limit(sys.maxsize)


def load_torch(name):
    Cls = getattr(torchtext.datasets,name)
    return Cls(root="data")

def main():
    """
    """
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        '--outdir',
        help = "write $outdir/$name.pkl")
    
    args = parser.parse_args()
    outdir = args.outdir

    # construct list of (name, lambda : data)
    DSS = []
    for name in (
            "AG_NEWS",
            "DBpedia",
            "YahooAnswers",
    ):
        DSS.append((name,lambda : load_torch(name)))

    DSS.append(('20News', load_20news))
    #ohsumed
    #R8
    #R52
    
    DSS.extend([
        ('kinnews', load_kinnews),
        ('kirnews', load_kirnews),
        ('filipino',load_filipino),
        ('swahili', load_swahili),
    ])
    name = "SogouNews"
    DSS.append((name,lambda : load_torch(name)))


    for name,fn in DSS:
        
        tr,te = fn()
    
        #unpack generators:
        tr = list(tr)
        te = list(te)
    
        outfile = os.path.join(outdir,name+".pkl")
        dtype = 'uint32'
        print(name,"tr,te:", (len(tr), len(te)))
        pickle.dump({
            'train_data': [t for (l,t) in tr],
            'test_data':  [t for (l,t) in te],
            'train_labels': np.array([l for (l,t) in tr],dtype),
            'test_labels':  np.array([l for (l,t) in te],dtype),
        },
                    open(outfile,'wb'))
        print("wrote:",outfile)

if __name__ == "__main__":
    main()
