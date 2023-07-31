# gzip-knn

Reimplentation of [npc_gzip](https://github.com/bazingagin/npc_gzip) of paper using `gzip + knn` for text classification.

**In progres...**

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

* prints top-1 accuracy, stores `--num_save=100` top nearest-neighbor indices so you can score using other `k`. So, the `outfile` is a `numpy` array of shape `(num_test, num_save)`, indexes (into training data) of the top `num_save` nearest neighbors.

## Notes

