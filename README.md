# gzip-knn

Reimplentation of [npc_gzip](https://github.com/bazingagin/npc_gzip) of paper using `gzip + knn` for text classification.

**In progres...**

## prepare data

```
```

## datasets info

`python datasets_info.py` should print,

```

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
time python compute.py --dataset kirnews.pkl --splitsize 10 --outfile /tmp/nn_kirnews.pkl --limit_train 250 --limit_test 50
```

Full `kirnews` dataset (~1 min on 8-core laptop):

```bash
time python compute.py --dataset kirnews.pkl --splitsize 10 --outfile /tmp/nn_kirnews.pkl --limit_train 250 --limit_test 50
```

* prints top-1 accuracy, stores `--num_save=100` top nearest-neighbor indices so you can score using other `k`.

## Notes

