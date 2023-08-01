"""
"""
import pickle
import os
from tabulate import tabulate
import numpy as np
from collections import Counter
import argparse

all_names1 = [
    'AG_NEWS', # 'AGNews'

    'DBpedia',
    'YahooAnswers',
    '20News',
    
    'ohsumed', # 'Ohsumed'
    'R8',
    'R52',
]

all_names2 = [
    'filipino',
    'kirnews',
    'kinnews',
    'swahili',
    'SogouNews',
]


def top_votes(counter):
    """
    Given a Counter object, return a list 
    of keys with the highest value.
    So, output len 1: no ties.
    output len > 1: these classes all tie.
    """
    top2 = counter.most_common(2)
    if len(top2) == 1:
        # only one class has any votes
        (cls,nvotes), = top2
        return [cls]
    else:
        # we have at least two classes with votes
        (cls1,n1),(cls2,n2) = top2
        assert(cls1 != cls2) #sanity check
        if n1 != n2:
            # not tied.
            assert(n1 > n2) #sanity check
            return [cls1]
        else:
            # n1==n2. top-2 are tied.
            return [cls for (cls,nn) in counter.items()]

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--k',
                        help="comma-separated k values",
                        default="1,2,3,4,5,11,21,35,51")

    parser.add_argument('--dir_nn',
                        required=True,
                        help="dir for nn files, $dir_nn/$name.pkl",
                        )

    parser.add_argument('--dir_data',
                        required=True,
                        help="dir for data files, $dir_data/$name.pkl",
                        )

    args = parser.parse_args()
    
    ks_str = args.k

    acc_fmt = "%.3f"
    
    ks = list(map(int,ks_str.split(",")))

    k2i = dict([(k,i) for (i,k) in enumerate(ks)])

    ks_set = set(ks)
    kmax = max(ks)

    results = []
    
    for names in (
            all_names1,
            all_names2,
    ):

        summary = []
        for name in names:

            f1 = os.path.join(args.dir_nn,  name+".pkl")
            f2 = os.path.join(args.dir_data,name+".pkl")
            
            nn = pickle.load(open(f1,'rb'))
            ds = pickle.load(open(f2,'rb'))
            ref = ds['test_labels']

            top_labels = ds['train_labels'][nn]
            n_test = len(ref)

            top2_correct = np.zeros((n_test,), 'bool')

            #
            # 'decrease' tie-breaking strategy
            #
            hyp_decrease = np.zeros((n_test,kmax),'uint32')

            #
            # 'rand' tie-breaking strategy
            #
            correct_rand = np.zeros((n_test,len(ks)),'float64')
            
            for i in range(n_test):
                refi = ref[i] # int
                hypi = top_labels[i] # [int]
                
                top2_correct[i] = (refi in hypi[:2])

                # count up from k=0,...,
                # accumulate in Counter, 'votes'.
            
                votes = Counter()

                for ki in range(kmax):
                    k = ki + 1
                    
                    votes[hypi[ki]] += 1

                    tv = top_votes(votes)
                    if len(tv) == 1:
                        # no ties
                        hyp_decrease[i][ki] = tv[0]
                    else:
                        # ties: len(tv) > 1
                        # - use previous value (k-1)
                        hyp_decrease[i][ki] = hyp_decrease[i][ki - 1]

                    if k in ks_set:
                        j = k2i[k]
                        if refi in tv:
                            # if in the tie set, random guess
                            # gives 1/c correct.
                            # (eg if len(tv)==1, 1/1=1.0, always right)
                            correct_rand[i][j] = 1.0 / len(tv)
                        
                        else:
                            # not in the tie-set: always wrong:
                            correct_rand[i][j] = 0.0


            table1 = []


            for k in ks:
                ki = k2i[k]

                acc_decr = (ref == hyp_decrease[:,ki]).mean()
                acc_rand = correct_rand[:,ki].mean()
                table1.append([
                        k,
                        acc_fmt % acc_decr,
                        acc_fmt % acc_rand,
                    ])

            hyp = ds['train_labels'][nn[:,0]]
            acc = (hyp==ref).mean()

            summary.append({
                'name': name,
                'knn1': acc,
                'top2': top2_correct.mean(),
            })

            results.append({
                'name': name,
                'k_table': table1,
            })

        rows = [
            ['knn1'] + [ (acc_fmt % a['knn1']) for a in summary],
            ['top2'] + [ (acc_fmt % a['top2']) for a in summary],
        ]
            
        print("")
        print(tabulate(
            rows,
            headers = [""] + [a['name'] for a in summary],
            disable_numparse=True))
        print("")

    for item in results:

        print("")
        print(item['name'])
        print(tabulate(
            item['k_table'],
            headers=['k',
                     'decr',
                     'rand'],
            disable_numparse=True))
        print("")
        
if __name__ == "__main__":
    main()
