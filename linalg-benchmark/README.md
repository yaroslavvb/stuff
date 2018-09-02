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

c5.9xlarge
numpy default        min:   294.77, median:   295.98, mean:   365.78
numpy gesvd          min:  1012.98, median:  1014.93, mean:  1015.95
numpy gesdd          min:   295.27, median:   296.04, mean:   296.12
TF CPU               min:  1115.32, median:  1118.19, mean:  1122.77
TF GPU               no GPU detected
PyTorch CPU          min:   867.44, median:   967.25, mean:   965.24
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

Results for i3.metal
numpy default        min:   266.03, median:   266.51, mean:   298.61
numpy gesvd          min:  1324.47, median:  1326.57, mean:  1327.88
numpy gesdd          min:   268.66, median:   269.29, mean:   269.50
TF CPU               min:  1086.69, median:  1090.78, mean:  1101.72
TF GPU               no GPU detected
PyTorch CPU          min:  1169.54, median:  1283.56, mean:  1279.05
PyTorch GPU          no GPU detected

Results for t3.2xlarge
numpy default        min:   993.98, median:  1010.18, mean:  1055.33
numpy gesvd          min:  3493.65, median:  3518.31, mean:  3517.02
numpy gesdd          min:   992.74, median:  1013.63, mean:  1010.28
TF CPU               min:  2118.30, median:  2159.09, mean:  2159.78
TF GPU               no GPU detected
PyTorch CPU          min:  2990.92, median:  3372.94, mean:  3385.87
PyTorch GPU          no GPU detected

Results for m5.24xlarge
numpy default        min:   353.14, median:   355.11, mean:   404.81
numpy gesvd          min:  1103.44, median:  1105.18, mean:  1105.80
numpy gesdd          min:   349.03, median:   350.66, mean:   350.80
TF CPU               min:  1156.68, median:  1160.73, mean:  1165.63
TF GPU               no GPU detected
PyTorch CPU          min:  1087.01, median:  1129.60, mean:  1143.58
PyTorch GPU          no GPU detected

```