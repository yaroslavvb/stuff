Linear algebra benchmark:

Need Python 3.6, pip, wget

Run standalone:
```
pip install -r https://raw.githubusercontent.com/yaroslavvb/stuff/master/linalg-benchmark/requirements.txt
wget https://raw.githubusercontent.com/yaroslavvb/stuff/master/linalg-benchmark/benchmark.py
python benchmark.py
```

Run on AWS:
```
pip install ncluster
python launch_aws.py
```

Results for SVD of 1534 x 1534 matrix:

```
2015 MacBook Pro
numpy default        min:   445.74, median:   480.14, mean:   491.55
numpy gesvd          min:  2123.99, median:  2551.53, mean:  2552.81
numpy gesdd          min:   584.27, median:   691.75, mean:   686.09
TF CPU               min:  2115.46, median:  2157.92, mean:  2228.17
TF GPU               no GPU detected
PyTorch CPU          min:  2468.93, median:  3316.59, mean:  3295.61
PyTorch GPU          no GPU detected

r5.large
numpy default        min:  1538.54, median:  1546.51, mean:  1573.60
numpy gesvd          min: 33868.77, median: 33902.69, mean: 33965.48
numpy gesdd          min:  1521.05, median:  1550.09, mean:  1549.20
TF CPU               ...
TF GPU               no GPU detected
PyTorch CPU          min:  3995.89, median:  4348.14, mean:  4364.10
PyTorch GPU          no GPU detected

c5.18xlarge
numpy default        min:   342.08, median:   343.94, mean:   346.90
numpy gesvd          min:   954.54, median:   956.79, mean:   958.16
numpy gesdd          min:   346.61, median:   348.00, mean:   348.26
TF CPU               min:  1165.11, median:  1170.42, mean:  1174.21
TF GPU               no GPU detected
PyTorch CPU          min:  1004.49, median:  1091.94, mean:  1087.53
PyTorch GPU          no GPU detected


p3.16xlarge
numpy default        min:   341.51, median:   342.65, mean:   408.34
numpy gesvd          min:  1264.59, median:  1265.48, mean:  1266.08
numpy gesdd          min:   341.22, median:   341.69, mean:   342.37
TF CPU               min:  1279.98, median:  1285.51, mean:  1292.32
TF GPU               min:  6962.91, median:  7006.48, mean:  8967.89
PyTorch CPU          min:  1048.54, median:  1226.51, mean:  1269.30
PyTorch GPU          min:   506.14, median:   511.30, mean:   513.09
```