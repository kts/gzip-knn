"""
currently outputs:

"""
from tabulate import tabulate
import pickle
import os
import argparse

long_names = {
    'kinnews': 'KinyarwandaNews',
    'kirnews': 'KirundiNews',
    'filipino': 'DengueFilipino',
    'swahili': 'SwahiliNews',
}

def percent(f):
    return "%0.1f%%" % (100.0 * f)

def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        '--datadir',
        help = "reads files datadir/$name.pkl")

    args = parser.parse_args()
    indir = args.datadir

    tab = []
    table2 = []
    for name in (

            "AG_NEWS",
            "DBpedia",
            "YahooAnswers",
            "20News",

            "ohsumed",
            "R8",
            "R52",
            
            'kinnews',
            'kirnews',
            'filipino',
            'swahili',

            "SogouNews",
    ):
        print(name)
        infile = os.path.join(indir,name+".pkl")
        if not os.path.isfile(infile):
            print("WARNING: MISSING: " + repr(infile))
            continue
        
        ds = pickle.load(open(infile,'rb'))
        
        n_train = len(ds['train_data'])
        n_test  = len(ds['test_data'])
        
        assert(ds['train_labels'].shape == (n_train,))
        assert(ds['test_labels'].shape == (n_test,))
        
        
        train_tuples = [(t,l) for (t,l) in zip(ds['train_data'],ds['train_labels'])]
        test_tuples  = [(t,l) for (t,l) in zip(ds['test_data'],ds['test_labels'])]
        
        train_set = set(train_tuples)
        test_set  = set(test_tuples)

        train_dups = n_train - len(train_set)
        test_dups  = n_test  - len(test_set)
        
        n_overlap = len(train_set.intersection(test_set))
        
        info = {
            'n_train': n_train,
            'n_test' : n_test,
        }
        
        tab.append((
            long_names.get(name,name),
            info['n_train'],
            len(train_set),
            percent(train_dups / n_train),
            
            info['n_test'],
            len(test_set),
            percent(test_dups / n_test),
        
            n_overlap,
            percent(n_overlap / len(test_set))
        ))
        
        n = 0
        for item in ds['train_data']:
            n += len(item)
        for item in ds['test_data']:
            n += len(item)
    
        table2.append((
            long_names.get(name,name),
            info['n_train'] / 1e3,
            info['n_test'] / 1e3,
            os.stat(infile).st_size / (2**20),
            
            (info['n_train'] * info['n_test'] * 8) / (2**30),
            
            n / (info['n_train'] + info['n_test']),
        ))
        
        
    headers = [
        "name",
        "tr","uniq","%dup",
        "te","uniq","%dup",
        
        "tr+te",
        "%",
    ]    
    print(tabulate(tab,
                   headers = headers))

    print("")

    print("")
    print(tabulate(table2,
                   headers = [
                       "name",
                       "train(K)",
                       "test(K)",
                       "len pkl(MB)",
                       "len dist mat(GB)",
                       "ave len data",
                   ]))

if __name__ == "__main__":
    main()
    
