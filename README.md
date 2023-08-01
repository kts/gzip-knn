# gzip-knn

Reimplentation of [npc_gzip](https://github.com/bazingagin/npc_gzip) of `gzip + knn` method for text classification. Paper: *“Low-Resource” Text Classification: A Parameter-Free Classification Method with Compressors* by Jiang et al. [link](https://aclanthology.org/2023.findings-acl.426/).

See my blog posts for context: [Part I](https://kenschutte.com/gzip-knn-paper/), [Part II](https://kenschutte.com/gzip-knn-paper2).

**In progress...**

## prepare data

The main code reads all the data from `pickle` files.  `prepare_data.py` loads data via the `data.py` file in the `npc_gzip` repo.

```
mkdir prepared
PYTHONPATH=/path/to/npc_gzip python prepare_data.py --outdir prepared
```

## datasets info

`python datasets_info.py --datadir prepared` should print,

```
name                  tr     uniq  %dup       te    uniq  %dup      tr+te  %
---------------  -------  -------  ------  -----  ------  ------  -------  ------
AG_NEWS           120000   120000  0.0%     7600    7600  0.0%          0  0.0%
DBpedia           560000   560000  0.0%    70000   70000  0.0%          0  0.0%
YahooAnswers     1400000  1400000  0.0%    60000   60000  0.0%          0  0.0%
20News             11314    11314  0.0%     7532    7532  0.0%          0  0.0%
ohsumed             3357     3357  0.0%     4043    4043  0.0%          0  0.0%
R8                  5485     5427  1.1%     2189    2176  0.6%          4  0.2%
R52                 6532     6454  1.2%     2568    2553  0.6%          6  0.2%
KinyarwandaNews    17014     9199  45.9%    4254    2702  36.5%       643  23.8%
KirundiNews         3689     1791  51.5%     923     698  24.4%       631  90.4%
DengueFilipino      4015     3951  1.6%     4015    3951  1.6%       3951  100.0%
SwahiliNews        22207    22207  0.0%     7338    7338  0.0%         34  0.5%
SogouNews         450000   450000  0.0%    60000   60000  0.0%          0  0.0%


name               train(K)    test(K)    len pkl(MB)    len dist mat(GB)    ave len data
---------------  ----------  ---------  -------------  ------------------  --------------
AG_NEWS             120          7.6        29.7412             6.79493           236.407
DBpedia             560         70         187.748            292.063             301.255
YahooAnswers       1400         60         737.988            625.849             520.806
20News               11.314      7.532      34.3779             0.634916         1902.53
ohsumed               3.357      4.043       9.06426            0.101122         1274.2
R8                    5.485      2.189       4.35275            0.0894566         585.777
R52                   6.532      2.568       5.55023            0.124977          630.411
KinyarwandaNews      17.014      4.254      39.0024             0.539255         1872.31
KirundiNews           3.689      0.923       7.79987            0.0253688        1721.47
DengueFilipino        4.015      4.015       0.534384           0.120105           62.737
SwahiliNews          22.207      7.338      62.5241             1.21411          2196.51
SogouNews           450         60        1357.86             201.166            2780.35
```

## Quick example

Run on subset of small dataset to make sure it's working (<2 seconds),

```bash
python compute.py --dataset kirnews.pkl --splitsize 10 --outfile /tmp/nn_kirnews.pkl --limit_train 250 --limit_test 50
```

Full `kirnews` dataset (~1 min on 8-core laptop):

```bash
python compute.py --dataset kirnews.pkl --outfile /tmp/nn_kirnews.pkl
```

* prints top-1 accuracy.

* stores `--num_save=100` top nearest-neighbor indices so you can score using other `k`. So, the `outfile` is a `numpy` array of shape `(num_test, num_save)`, indexes (into training data) of the top `num_save` nearest neighbors.

## Results

```bash
# reads:
#  $dir_nn/$name.pkl and
#  $dir_data/$name.pkl
python score.py --dir_nn nn --dir_data prepared > results.txt
```

Prints summary,

```
      AG_NEWS    DBpedia    YahooAnswers    20News    ohsumed    R8     R52
----  ---------  ---------  --------------  --------  ---------  -----  -----
knn1  0.876      0.942      0.485           0.607     0.365      0.913  0.852
top2  0.937      0.970      0.622           0.685     0.481      0.952  0.889


      filipino    kirnews    kinnews    swahili    SogouNews
----  ----------  ---------  ---------  ---------  -----------
knn1  0.999       0.858      0.835      0.850      0.951
top2  1.000       0.906      0.891      0.927      0.973
```

and prints details for different k for each dataset. For example, one dataset

```
AG_NEWS
k    decr    rand
---  ------  ------
1    0.876   0.876
2    0.876   0.863
3    0.891   0.888
4    0.892   0.889
5    0.895   0.889
11   0.896   0.893
21   0.896   0.889
35   0.898   0.883
51   0.898   0.878
```

`decr` and `rand` are two approaches to breaking ties in `kNN`

* `desr`: decrease `k` until there is no tie.

* `rand`: randomly select one of the ties. This this computed by giving a score of `1/num_ties` - i.e. taking the expected value.


See [results.txt](results.txt) for all.

