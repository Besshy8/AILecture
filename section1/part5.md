# Pythonを用いた初心者向けAI実践講座(中級編) 11/9 配布資料

## 1-6 教師なし学習

　教師データを使わずに学習を行うことを「教師なし学習」と呼ぶ。教師なし学習は、
主にクラスタリングの用途で使われ、データに隠れている構造を発見したり、
教師あり学習の前処理として用いられる。

### 1-6-1 k-meansアルゴリズム

　クラスタリング(教師なし学習)によく使われる手法に「K-meansアルゴリズム」がある。
K-meansアルゴリズムでは以下の目的関数を最小化する。

$$J = \sum_{n=1}^{N} \sum_{k=1}^{K} r_{nk}||\vec{x}_n - \vec{\mu}_n||^2 \tag{1.6.1}$$

$\vec{\mu}_k$について解くと、

$$\vec{\mu}_k = \frac{\sum_nr_{nk}\vec{x}_n}{\sum_nr_{nk}} \tag{1.6.2}$$

　この式の分母は$k$番目のクラスタに割り当てられたデータの数に等しいので、$\vec{\mu}_k$は、
$k$番目のクラスタに割り当てられた全てのデータ点$\vec{x_n}$の平均となっている。これがK-meansアルゴリズムと呼ばれている理由である。
(なお、K-meansアルゴリズムは次章の混合ガウス分布に対するEMアルゴリズムの非確率的
極限となっている。)

\clearpage

(k-means法)

\clearpage

### 1-6-2 次元削減

　教師なし学習の応用の一つに「次元削減」があげられる。学習するパラメータ
の数を減らすことができるため、計算時間を減らすために主に用いられる。

　次元削減は数学的に見れば、データ空間でより次元が低い部分空間にデータ点を
射影することで実現される。しかしその過程でデータが元々持っていた情報が
失われてしまうため、できるだけ情報を残しながらデータ空間の次元を下げる
必要がある。これを実現するための手法として、データの持つ分散(ばらつき)
が大きくなるようにデータを射影するものがあり、これは「主成分分析」と呼ばれる。

　学習用データを表す行列を$X(N\times M)$とおく。このときデータの共分散行列は以下の
ように表すことができる。

$$\Sigma = \frac{1}{N}X^TX \tag{1.6.3}$$

　ある特定の方向をもつ単位ベクトル$\vec{e}$を考える。このときこの単位ベクトルにデータ点を射影してできる新しいデータ点に対して分散$\sigma$を考えると、

$$\sigma = \vec{e}^T\Sigma\vec{e} \tag{1.6.4}$$

となる。この$\sigma$に対して、ラグランジュ未定乗数法を適用して$\sigma$
の最大値を求める(分散最大化)をしようとすると、以下の最適化問題を解く
ことになる。

$$argmax_{\vec{e}}\Biggl\{L(\vec{e},\lambda) = \sigma - \lambda(\vec{||e||^2} -1)\Biggl\} , \quad \vec{||e||^2} = 1 \tag{1.6.5}$$

この最適化問題を解くと主成分分析は以下の固有値問題、

$$\Sigma \vec{e} = \lambda \vec{e} \tag{1.6.6}$$

となる。分散が最大になるような単位ベクトル$\vec{e}$を「第一主成分」と呼び、
上記の固有値問題のうち、固有値が最大になる固有ベクトルが第一主成分
に対応する。

(1)3つの2次元データ$\vec{a_1},\vec{a_2},\vec{a_3}$を考え、主成分分析が式(1.6.6)の固有値問題で定式化されることを確認せよ。
 
(略解) $\vec{a_1},\vec{a_2},\vec{a_3}$を並べた3×2行列$X$を考え、

$$\Sigma = \frac{1}{3}X^TX$$

が成立することを確認する。次に2次元単位ベクトル$\vec{e}=(e_x,e_y)^T$
を考え、$\vec{a_1},\vec{a_2},\vec{a_3}$それぞれとの内積を考えること
で分散$\sigma$を求めて式(1.6.4)を示す。最後にこの$\sigma$を用いて式(1.6.5)を
構成してラグランジュ未定乗数法を用いて以下の値を計算してその結果を用いて
固有方程式(1.6.6)を導出する。(補足　2次形式の標準形と最大最小の議論(1-1-13節)について知っていれば、
式(1.6.4)を導出した時点で直ちに固有値を求めに行ってもよい。)

$$\frac{\partial L(\vec{e},\lambda)}{\partial \vec{e}},\quad\frac{\partial L(\vec{e},\lambda)}{\partial \lambda}$$

(解答)

\clearpage

(解答)
