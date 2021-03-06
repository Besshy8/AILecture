# 第２章　ベイズ機械学習

## 2-1 確率統計の基礎

　第2章では、ベイズ機械学習を取り扱う。あまり聞き慣れない言葉かもしれないが、PRML等
の機械学習の名著を読む際には必須の知識であり、応用においても、近年の話題の深層学習(DeepLearning)と
双璧をなすツールである。(これら2つの間の密接な関係を示した論文も多く存在している。)
まず本節では、ベイズ機械学習の前提知識となる「確率統計」という数学の分野を扱う。


### 2-1-1 事象族、確率、確率分布

確率的に取り扱いたいものがあったとして、そのシステムを次のように表現するとする。

$$\Omega = \{w_1,w_2,...\}$$

$\Omega$は標本空間、$w_1,...$は根元事象と呼ばれている。非常に抽象的だが、サイコロを例として考えると、サイコロを1回ふるという問題を考えるときは、次のような表記をすると単に言っているだけである。

$$\Omega = \{w_1,...,w_6\} = \bigl\{1が出る, ... , 6が出る\bigl\}$$

次にサイコロの目の出方に対して何に注目するか(これをここでは興味と呼ぶことにする)を考えて、以下のような表を作成する。

(興味1) 奇数?or偶数?

|$F'$|$\{w_1,w_3,w_5\}$|$\{w_2,w_4,w_6\}$|
|:----:|:----:|:----:|
|$P$|1/2|1/2|

:サイコロの確率分布1

$$F' = \{\ \{w_1,w_3,w_5\}, \ \{w_2,w_4,w_6\}\ \}$$

この事象族$F'$の要素(集合)に対してある数値を返すものを「確率」と呼ぶ。(集合から数値への写像とも言える。$P\bigl(\{w_1,w_3,w_5\}\bigl) = 1/2$)
また、「事象族$F'$」と「確率$P$」を合わせたものを「確率分布」と呼ぶ。(要するに表1が確率分布である。)よって興味が変われば確率分布もそれに対応して変化する。

(興味2) 1?or1以外?

|$F'$|$\{w_1\}$|$\{w_2,w_3,w_4,w_5,w_6\}$|
|:----:|:----:|:----:|
|$P$|1/6|5/6|

:サイコロの確率分布2


(1)赤玉7個、白玉3個入っている箱から、玉を2回取り出す。(1回玉を取り出してから、それを箱に「戻し」、もう一度取り出す。)

1. 標本空間$\Omega$を求めよ。

2. 「玉の色の組み合わせ」に興味がある場合の事象族$F_1$を定め、確率分布を求めよ。

3. 「白玉が出る or 出ない」に興味がある場合の事象族$F_2$を定め、確率分布を求めよ。

### 2-1-2 確率変数

次のような確率分布を考える。

|$F'$|$\cdots  A_i \cdots$|
|:-----:|:-----:|
|$P$|$\cdots  p_i \cdots$|

:一般的な確率分布$P(A_i) = P(\{w\in A_i\}) = p_i$

この分布において、$w\in A_i$のとき($A_i$が起こったとき)、数値$x_i$が定まるという状況を考える。つまり、

$$X(w) = \begin{cases}
            x_1 & (w\in A_1) \\
            \ \vdots & \qquad\vdots \\
            x_n & (w \in A_n)
        \end{cases}$$

である。この$X$を確率変数と呼ぶ。(ここで示したように、確率変数は厳密には、根元事象の関数である。)確率変数を含めた表は以下のようになる。

|$F'$|$\cdots  A_i \cdots$|
|:-----:|:-----:|
|$X$|$\cdots  x_i \cdots$|
|$P$|$\cdots  p_i \cdots$|

具体的な例は、次節の練習問題で触れる。

### 2-1-3 期待値、分散

期待値(平均)を次のように定義する。

$$E(X) = \sum_{i=1}^{n}x_ip_i$$

この量が実際に妥当なものなのかを次の問題で考える。

(2)次の確率分布について考える。

|$F'$|赤|青|黄|
|:-----:|:-----:|:-----:|:-----:|
|$X$|$10$|$100$|$1000$|
|$P$|$6/10$|$3/10$|$1/10$|

:確率分布

以下のようなケースを考え、式()で定義した平均の妥当性を調べなさい。(詳細未定)

　分散については次のように定義する。

$$Var(X) = E\bigl[(X-E[X])^2\bigl]$$

また、単位を確率変数$X$に揃えたものを標準偏差といい、分散を用いて

$$\sqrt{Var(x)} = \sqrt{E\bigl[(X-E[X])^2\bigl]}$$

と定義する。

(3)分散、標準偏差の計算 (未定)

### (重要) 期待値、分散の性質1

1. $E[aX + b] = aE[X] + b$

2. $Var[aX + b] = a^2Var[x]$

3. $Var[X] = E[X^2] - E[X]^2$


### 2-1-4 同時分布、周辺確率

ある対象の2つの事象族$F_A,F_B$を考える。

$$F_A = \{A_1,A_2,\cdots,A_n\}$$
$$F_B = \{B_1,B_2,\cdots,B_m\}$$

この事象族に対して以下のような確率分布を考えたとき、この確率分布を「同時(確率)分布」と呼ぶ。

|$F_B \ / \ F_A$|$A_1\qquad \cdots\cdots \qquad A_n$|
|:-----:|:---------:|
|$B_1$|$P(A_1\cap B_1)\cdots\cdots P(A_n \cap B_1)$|
|$\vdots$||
|$B_n$|$P(A_1\cap B_m)\cdots\cdots P(A_n\cap B_m)$|

:同時分布$P_{AB}(A_i\cap B_j)$

例えばサイコロの同時分布の例として以下のようなものを考えることができる。

|$F_B / F_A$|$\{偶数\}$|$\{奇数\}$|
|:-----:|:-----:|:-----:|
|$\{3以下\}$|$1/6$|$2/6$|
|$\{4以上\}$|$2/6$|$1/6$|

:同時分布の例

同時分布がわかれば、対象に対してあらゆる情報を持っているということになる。例えば、表5に対して$i$行目の要素の和をとり、

$$P_{AB}(A_1\cap B_i) + P_{AB}(A_2\cap B_i) + \cdots + P_{AB}(A_n\cap B_i) $$
$$=P_{AB}\bigl(\{A_1\cap B_i\}\cup\{A_2\cap B_i\}\cup...\cup\{A_n\cap B_i\}\bigl)$$
$$=P_{AB}(B_i)\equiv P_B(B_i)$$

とすることができる。これを「周辺分布」と呼び、以下のように定義され、このような操作を「周辺化」と呼ぶ。

$$P_A(A_i) \equiv \sum_{j=1}^{m}P_{AB}(A_i\cap B_j)$$
$$P_B(B_j) \equiv \sum_{i=1}^{n}P_{AB}(A_i\cap B_j)$$

表5に周辺分布の情報を入れると以下のようになる。

|$F_B \ / \ F_A$|$A_1\qquad \cdots\cdots \qquad A_n$||
|:-----:|:---------:|:-----:|
|$B_1$|$P(A_1\cap B_1)\cdots\cdots P(A_n \cap B_1)$|$P_B(B_1)$|
|$\vdots$|||
|$B_n$|$P(A_1\cap B_m)\cdots\cdots P(A_n\cap B_m)$|$P_B(B_m)$|
||$P_A(A_1)\quad \cdots\cdots \quad P_A(A_n)$|

:同時分布と周辺分布

(補足)
後ほど出てくるが、例えば機械学習であれば、$p(X,y,w)$といった同時分布を学習の際に用いて(構築して)、予測する際は学習に用いた重み$w$を周辺化によって同時分布から削除するといった操作をよく行う。このように注目している確率変数以外のものを周辺化によって消去する操作として周辺化は有用である。(ちなみに、実際は同時分布$p(X,y,w)$を構築することは少なく、ベイズの定理を
用いて学習則を記述することがほとんどである。しかし、同時分布には対象の情報が
全て含まれているわけなので、これを求めたいというモチベーションは当然機械学習で応用されても同じである。同時分布を直接構築する方法はベイズ機械学習の分野では「生成モデル」と呼ばれていて、これを求められれば、学習データから未知のデータを生成するといったことが可能になる。生成モデルについては2-3節、確率的生成モデルを参照。)


### 2-1-5 独立性

以下の性質が成り立つとき、$F_A, F_B$は「独立」と呼ばれている。

$$P_{AB}(A_i\cap B_j) = P_A(A_i)P_B(B_j)$$

(3)1つのサイコロを続けて2回投げる試行を行う。以下のように事象族$F_1,F_2$を定めるとき、2つの事象族が独立であるかどうかを判定せよ。

$F_1 = \{A_1,A_2\}, F_2 = \{B_1,B_2\}$とする。

1. $A_1 = \{1回目のサイコロの目が偶数\}, A_2 = \{1回目のサイコロの目が奇数\} \\ B_1 = \{1回目と2回目で同じ目\}, \quad \quad B_2 = \{1回目と2回目で違う目\}$

2. $A_1 = \{1回目のサイコロの目が偶数\}, A_2 = \{1回目のサイコロの目が奇数\} \\ B_1 = \{2回の試行で最低1回1が出る\}, B_2 = \{1回も1が出ない\}$


### (重要) 期待値、分散の性質2

1. $E_{XY}[X+Y] = E_X[X] + E_Y[Y]$
2. $V_{XY}[X+Y] = V_{X}[X] + V_Y[Y] + 2(E_{XY}[XY] - E_X[X]E_Y[Y])$

性質2の右辺第3項目は「共分散」と呼ばれており、以下のように定義される。

$$Cov[X,Y] = E_{XY}[XY] - E_X[X]E_Y[Y]$$

上記の独立性の性質を使うと、

$$E_{XY}[XY] = E_X[X]E_Y[Y]$$

が成立することがわかるため、$Cov_{XY}[X,Y]=0$となる。このような状態を
「$X$と$Y$は無相関である」という。(注意 独立なら無相関である。無相関なら独立なのではない。)

(補足) 上記説明は結構乱暴なもので、これらの性質を厳密に示していくには、
多次元確率分布の定義をして、性質を見ていかないといけない。その中でいくつかの重要な事項(確率分布の畳み込みや大数の法則)が導かれるのでそれはColumnにゆずった。ここでは、分散には単純な線形性が成り立たないことと、
共分散を抑えておけば十分である。

### 2-1-6 条件付き確率

表7を再掲する。

|$F_B \ / \ F_A$|$A_1\qquad \cdots\cdots \qquad A_n$||
|:-----:|:---------:|:-----:|
|$B_1$|$P(A_1\cap B_1)\cdots\cdots P(A_n \cap B_1)$|$P_B(B_1)$|
|$\vdots$|||
|$B_j$|$P(A_1\cap B_j)\cdots\cdots P(A_n \cap B_j)$|$P_B(B_j)$|
|$\vdots$|||
|$B_n$|$P(A_1\cap B_m)\cdots\cdots P(A_n\cap B_m)$|$P_B(B_m)$|
||$P_A(A_1)\quad \cdots\cdots \quad P_A(A_n)$|

:同時分布と周辺分布

この表において$B_j$が起こったとき($B_j$の情報が得られたとき)、次のような確率を考え、これを「条件付き確率」と呼ぶ。

$$P(A_i|B_j) \equiv \frac{P(A_i\cap B_j)}{P(B_j)}$$

(こうして定義した量が確率分布の性質を満たしているかは別途証明が必要だが、
そこまで難しくないので割愛。)

また、$A$と$B$が独立なとき、$P(A_i\cap B_j)=P(A_i)P(B_j)$なので、

$$P(A_i|B_j) = \frac{P(A_i\cap B_j)}{P(B_j)} = \frac{P(A_i)P(B_j)}{P(B_j)} = P(A_i)$$

となる。(この性質から、独立性の定義が妥当だったとも言える。)

(4)奇数の目の部分に黒いシールが、偶数の目の部分に白いシールが貼られているサイコロを1つ投げる試行を考える。このサイコロを投げて「黒」が出たときの確率分布を考えなさい。


### 2-1-7 ベイズの定理

2-1-6で定義した条件付き確率$P(A_i|B_j)$は、「確率$P(A_i)$が、$B_j$が与えられたことによって、$P(A_i|B_j)$に更新された」と捉えることができる。式()に戻り、条件付き確率を少し変形すると、

$$P(A_i|B_j) = \frac{P(A_i\cap B_j)}{P(B_j)} = \frac{P(B_j\cap A_i)}{P(B_j)} = \frac{P(B_j|A_i)P(A_i)}{P(B_j)}$$ 
$$ = \frac{P(B_j|A_i)}{P(B_j)}P(A_i)$$

となり、これを「ベイズの定理」という。これは上記の、「確率$P(A_i)$が、$B_j$が与えられたことによって、$P(A_i|B_j)$に更新された」という表現を数式の上で明らかにするために、$P(A_i)$を陽に表した式変形とも言える。
後の章で扱うが、この$P(A_i)$を「事前分布」、更新された$P(A_i|B_j)$を「事後分布」と呼ぶ。ベイズ機械学習は単に、このベイズの定理を使って事前分布の情報を更新し、データに対する予測精度を上げていく手法であり、この更新プロセスを「学習」と呼んでいる。このベイズの定理を用いた「知識(情報)の更新」は応用上非常に強力で、機械学習のみならず様々な工学分野で応用されている。

ちなみに式()に現れている$P(B_j|A_i)$は、この式の中では、「尤度関数」と呼ばれている。また、さらに$P(A_i)$を陽に表すために、周辺分布の性質を使って以下のような表現をしてあることが多い。

$$P(A_i|B_j) = \frac{P(B_j|A_i)}{P(B_j)}P(A_i) = \frac{P(B_j|A_i)P(A_i)}{\sum_{i}P(B_j|A_i)P(A_i)}$$

(ただし、式変形した結果からわかるように、単にこれは分子の値を足し合わせて規格化しているだけなので、実際は特に気にせず分子だけ計算して問題に合わせて全確率の性質を使って規格化すればいい。)

(5)ある海域に船(潜水艦)が沈没している。この船を引き揚げるために、以下のような方策をとる。まず、その海域をグリッドに分割し、各グリッドに船が沈んでいる確率を適当に割り振る。次に、実際に1つのグリットにおいて船の捜索を行い、
もし船が見つからなかったら、そのグリッドに船が沈んでいる確率を更新し、
同様に、他のグリッドに船が沈んでいる確率を(全確率の和が1であるように)均等に変化させる。あるグリットに注目し、

$$A = \{そのグリッドに船が沈んでいる\}$$
$$B = \{そのグリッドにおける捜索の結果、船が発見される\}$$

という2つの事象を考える。さらに$p=P(A), q = P(B|A)$は予め決まっているものとする。このとき、上記の確率の更新則を決めなさい。


### 2-1-8 確率密度関数

連続的な確率分布を考える(例 正規分布)。この場合の標本空間$\Omega$は、

$$\Omega = \{R(実数)上に値をとる\}$$

この場合の$F',P$を次のように考える。連続的な値をとる場合はとりうる事象が
無限通りになって議論ができないので、(微小)区間$\Delta x$を考え、事象族$F'$を、

$$F' = \{\cdots,\{[x + \Delta x) にある\},\cdots \}$$

とする。さらに、これに対する確率$P$を次のように決める。

$$P\bigl(\{[x + \Delta x)\}\bigl) = p(x)\Delta x$$

つまり確率分布は、

|$F'$|$\cdots ,\{[x + \Delta x)\}, \cdots$|
|:----:|:----:|:----:|
|$P$|$\cdots ,p(x)\Delta x, \cdots$|


この$p(x)$を確率密度関数と呼ぶ。(注意 確率$P$と確率密度関数$p(x)$は上記のように別物である。)

上記のような定義から、確率密度関数は次のような性質が要請される。

1. $p(x) \geq 0$

2. $P\bigl(\{[a,b]\}\bigl) = \int_{a}^{b}p(x)dx$

3. $\int_{-\infty}^{\infty}p(x)dx = 1$ (規格化条件)


### 2-1-9 有名な確率密度関数

ベイズ機械学習においてよく出る確率分布の確率密度関数についていくつか紹介する。

### ベルヌーイ分布

$$ Bern(x) = \mu^x(1-\mu)^{1-x}$$

この分布の平均と分散は、

$$E[x] = \mu$$
$$var[x] = \mu(1-\mu)$$

### カテゴリー分布

$$Cat(\vec{s}) = \prod^{K}_{k=1}\pi_k^{s_k}$$

$$E[s_k] = \pi_k$$
$$var[x] = \mu(1-\mu)$$


### ポアソン分布

$$Poi(x) = \frac{\lambda^x}{x!}e^{- \lambda}$$

$$E[x] = \lambda$$
$$var[x] = \lambda$$


### 正規分布(ガウス分布)

$$p(x) = \frac{1}{\sqrt{2\pi \sigma^2}}\exp\biggl[-\frac{1}{2\sigma^2}(x-\mu)^2\biggl]$$

$$E[x] = \mu$$
$$var[x] = \sigma^2$$


### (おまけ) ベータ分布

$$ Beta(\mu) = \frac{\Gamma(a + b)}{\Gamma(a)\Gamma(b)}\mu^{a-1}(1-\mu)^{b-1}$$


$$E[x] = \frac{a}{a+b}$$
$$var[x] = \frac{ab}{(a+b)^2(a+b+1)}$$


### (おまけ2) スチューデントの$t$分布

$$St(x) = \frac{\Gamma(v/2 + 1/2)}{\Gamma(v/2)}\biggl(\frac{\lambda}{\pi v}\biggl)^{1/2}\biggl[1 + \frac{\lambda(x-\mu)^2}{v}\biggl]^{-v/2-1/2}$$


(6)確率密度関数が、

$$p(x) = \frac{1}{\sqrt{2\pi \sigma^2}}\exp\biggl[-\frac{1}{2\sigma^2}(x-\mu)^2\biggl]$$

で表される正規分布について、平均と分散を求めよ。



\clearpage

## 2-2 ベイズ推論の基礎

### 2-2-1 ベイズ推論における学習、予測

ベイス推論では、パラメータ（重み）も不確実性をもつ確率変数として捉え、
次の手順で問題を解くことが多い。

1. 問題に合わせて、適切な尤度関数$p(D|\vec{w})$を設定する。（モデル化）
2. 尤度関数にあった共役事前分布$p(\vec{w})$を選ぶ。
3. ベイスの定理を用いて、事後分布$p(\vec{w}|D)$を解析的に求める。（学習）
4. 事後分布を用いて、予測分布$p(x_*|D)$を計算する。（予測）

### $Step1.$　尤度関数$p(D|\vec{x})$の設計（モデル化）
解きたい問題に対して可視化などを行い、ある程度妥当であると思われる尤度関数を設定する。
線形回帰など一般的にはガウス分布を用いることが多いが、カウントデータ（非負）ならポアソン分布、周期性をもつ分布はフォンミーゼス分布（PRML2章）など、
データの分布に合わせた確率分布を選ぶことが望ましい。

### $Step2.$　共役事前分布$p(\vec{w})$の設定
設定した尤度関数の共役事前分布$p(\vec{w})$を選ぶ。「共役事前分布」はベイズの
定理ととても相性がよく、ベイズの定理を適用してもその分布の形が変わらない
分布である。大抵は尤度関数と1対1の関係で決まっているので、この$Step2.$は
すぐに終わる。

### $Step3.$ 　学習
ベイスの定理を用いて以下の事後分布$p(\vec{w}|D)$を計算する

$$p(\vec{w}|D)=\frac{p(D|\vec{w})p(\vec{w})}{p(D)}$$

### $Step4.$　 予測
未観測のデータ$x_*$に対して以下の予測分布を計算する。

$$p(x_*|D)=\int p(x_*|\vec{w})p(\vec{w}|D) \ d\vec{w}$$

これは予測に際して必要ない$\vec{w}$について積分除去を行ったものと考えることができる。
また、事後分布とは異なり、一般的には予測分布は共役事前分布の形になるとは
限らない。

（1）尤度関数としてベルヌーイ分布

$$p(x|\mu)=Bern(x|\mu)$$

でモデル化できる問題において、$\mu$の分布を訓練データ$x_n$から推論
せよ。また未観測の値$x_*\in 0,1$に対する予測分布を計算せよ。

（2）線形回帰  $y_n=\vec{w}^Tx_n + \epsilon_n$ についてモデル$p(y_n|\vec{x}_n,\vec{w})$の
構築を行い、事後分布、予測分布を計算せよ。

### 2-2-2 モデルエビデンス（周辺尤度）
ベイズの定理を変形して、

$$p(D)=\frac{p(D|\vec{w})p(\vec{w})}{p(\vec{w}|D)}$$

と表す。このとき、$p(D)$を周辺尤度（モデルエビデンス）と呼ぶ。これは
モデルのデータ生成確率と解釈することができ、この値を複数のモデル間で
比較することで最適なモデルの選択を行うことができる。


## 2-3 確率的生成モデル
現実の問題では、データを生成する分布は複雑で1つの確率分布で取り扱えるケースは多くない。複数の分布をデータの生成過程を仮定しながら組み合わせて全体のモデル
（同時分布）を作り、そこから事後分布、予測分布を計算する手法を「確率的
生成モデル」と呼び、確率分布を複数組み合わせてできたモデルを「混合モデル」と呼ぶ。

### 2-3-1 混合モデルの構築

多峰性をもつデータに関してのクラスタリングを考える。データを表現するため
のモデルを構築する要件定義として例えば以下の過程を考える。

1. K個のクラスタは混合比率$\pi=(\pi_1,...,\pi_K)$で分布上に存在し、
$\pi$は事前分布$p(\pi)$から生成される。
2. それぞれのクラスタ自身の持つパラメータ$\theta_k$が事前分布$p(\theta_k)$から生成される。
3. データ点$x_n$が$K$個ある分布うちのどれかから生成されるとし、
$x_n$に対応するクラスタの割り当てを$s_n$をする。この$s_n$は
比率$\pi$によって決まるとし、$s_n$の生成する分布を
$p(s_n|\vec{\pi})$とする。

4. $s_n$によって選択された$k$番目の確率分布
$p(\vec{x_n}|\vec{\theta_k})$からデータ$x_n$が生成される。

これら全ての確率分布をデータ生成順に組み合わせ、$N$個のデータに関して同時分布を考えると以下のようになる。

$$p(X,S,\Theta,\vec{\pi}) = p(X|S,\Theta)p(S|\vec{\pi})p(\Theta)p(\vec{\pi}) 
= \bigl\{\prod_{n=1}^{N}p(\vec{x_n}|\vec{s_n},\Theta)p(\vec{s_n}|\vec{\pi})\bigl\} \bigl\{\prod_{k=1}^{K}p(\vec{\theta_k})\bigl\}p(\vec{\pi})$$

実際に問題を解く際には、$p(X|S,\Theta)$,$p(\Theta)$は問題設定に
応じて決め、（クラスタリングの場合は）$s_n$をサンプリングする分布として
以下のカテゴリ分布、

$$p(\vec{s_n}|\vec{\pi}) = Cat(\vec{s_n}|\vec{\pi}) = \prod_{k=1}^K \pi_{k}^{s_{n,k}} $$

$\pi$をサンプリングする分布としてカテゴリ分布の共役事前分布である
$Dirichlet$分布を選ぶことが多い。

$$p(\vec{\pi})=Dir(\vec{\pi}|\vec{\alpha})$$ 

また$s_n$は直接は観測されないが、$x_n$を生成する$K$個の分布のうち1つを
選択するという意味で、$x_n$を発生させる確率分布を潜在的に決めている
確率変数であると言える。このため$s_n$は潜在変数と呼ばれている。

（1）あるクラスタ$k$に対する観測モデルとしてポアソン分布を採用し、
混合モデルを構築せよ。

### 2-3-2 混合モデルの推論

この同時分布から事後分布$p(S,\Theta,\vec{\pi}|X)$,クラスタ$S$の推定$p(S|X)$が可能であるが、いずれの計算も

$$p(X)=\sum_{S}\iint p(X,S,\Theta,\pi) d\Theta d\pi \\
=\sum_{s}p(X,S)$$

$$p(S|X) = \iint p(S,\Theta,\pi|X) d\Theta d\pi$$

の計算が発生してしまい、解析的に解くことがほぼ不可能になる。次章で、この問題
をある程度解消して近似的に解を出す方法を説明する。



## 2-4 近似推論

事後分布、周辺尤度、予測分布など問題によっては解析的に解くことが難しい
ものに関しては、近似的に解を求めることが多い。近似手法は大きく分けると、サンプリング、変分法に大別される。

### 2-4-1 ギブスサンプリング

分布全体の解析的な把握が難しい場合、期待値等の分布に関する部分的な統計
量を解析することは重要である。そのような各種統計量を得たい場合、分布
から複数の実現値をサンプリングし、その実現値を元に計算を行うことが
有効的である。

$$z_1^{(i)},z_2^{(i)},z_3^{(i)} \sim \ p(z_1,z_2,z_3)$$

混合モデル等、複雑なモデルに関しては全てのサンプルを上記のように同時に
サンプルすることは難しいため、ギブスサンプリングという手法を用いて以下の
ようにサンプリングを行う。

$$z_1^{(i)} \sim \ p(z_1|z_2^{(i-1)},z_3^{(i-1)})$$
$$z_2^{(i)} \sim \ p(z_2|z_1^{(i)},z_3^{(i-1)})$$
$$z_3^{(i)} \sim \ p(z_3|z_1^{(i)},z_2^{(i)})$$

この手法はMCMC（マルコフ連鎖モンテカルロ法）の手法の一つに分類されており、
サンプル数が十分に多い場合、繰り返しで得られた$z_k$は真の事後分布から
得られたものであると理論的に保証されている。（$Column$参照）


（1）ギブスサンプリングを用いて、（）で求めたポアソン混合モデルの
事後分布$p(S,\vec{\lambda},\vec{\pi}|X)$からサンプリングを行う
アルゴリズムを導け。混合分布では以下のように、潜在変数とパラメータを次の
ように分けてサンプリングすると簡単な確率分布が得られることが知られている。

$$S \sim p(S|X,\vec{\lambda},\vec{\pi}), \quad \vec{\lambda}, \ \vec{\pi} \sim p(\vec{\lambda},\vec{\pi}|X,S)$$

### 2-4-2 平均場近似(変分推論)

複雑な分布を最適化問題を解くことによってより簡単な近似分布で表現する手法を「変分推論」、「変分近似」と呼ぶ。事後分布は解析的に解けなくなる状況に陥ることがあるため、確率変数に特定の制約を付けた上で事後分布を近似する。

最適化にはKLダイバージェンスを使い、最小化問題として以下のように定式化される。


$$q_{opt} = argmin_q KL[q(z_1,z_2,z_3) | \ p(z_1,z_2,z_3)]$$

ここで、解が$q_{opt}(z_1,z_2,z_3) = p(z_1,z_2,z_3)$とならないように$q$に制約
をつける手法として、各確率変数に独立性の仮定をおく。

$$p(z_1,z_2,z_3) \approx q(z_1)q(z_2)q(z_3)$$

これを「平均場近似」と呼ぶ。

（2）平均場近似を用いて、（）で求めたポアソン混合モデルの変分推論アルゴリズム
を導出せよ。ただし、事後分布$p(S,\vec{\lambda},\vec{\pi}|X)$
の潜在変数とパラメータを以下のように分けて近似せよ。

$$p(S,\vec{\lambda},\vec{\pi}|X) \approx q(S)q(\vec{\lambda},\vec{\pi})$$


## 2-5 ガウス混合モデルと教師なし学習

### 2-5-1 潜在変数とガウス混合モデル 

2-3節からわかるように、複雑なモデルの定式化の際に「潜在変数」を取り入れること
で問題が簡単になることがある。潜在変数を確率モデルの中に取り入れることを陽に
表すと次のように定式化される。

$$p(x) = \sum_{z}p(x,z) =\sum_{z}p(x|z)p(z) $$

式()を用いて混合ガウス分布の密度関数$p(x)$を求めると、

$$p(x) = \sum_{z}p(x|z)p(z) = \sum_{k=1}^{K}\pi_kN(x|\mu_k,\Sigma_k)$$

となる。

(1) 混合ガウス分布の密度関数$p(x)$が式()で表されることを示せ。

### 2-5-2 EMアルゴリズム(Expectation-Maximization Algorithm)

潜在変数が含まれる最尤推定の問題で使われる最適化アルゴリズム。

2-5-1で求めた$p(x)$の対数尤度関数は以下のようになる。

$$\ln p(X|\pi,\mu,\Sigma) = \sum_{n=1}^{N}\ln\Bigl\{\sum_{n=1}^{N}\pi_k N(x_n|\mu_k,\Sigma_k)\Bigl\}$$

このような潜在変数が含まれる尤度関数の最尤推定では、「EMアルゴリズム」と呼ばれる手法を使うと効率よく解を求められることが知られている。
EMアルゴリズムは以下の4つのステップからなる。

1. $\mu_k,\Sigma_k,\pi_k$を初期化し、対数尤度()の初期値を計算する。
2. (Eステップ) 1の値を用いて以下の「負担率」を計算する。

$$\gamma_k = \frac{\pi_k N(x_n|\mu_k,\Sigma_k)}{\sum_{j=1}^{K}\pi_j N(x_n|\mu_j,\Sigma_j)}$$

3. (Mステップ) 2で求めた負担率を用いて、次式で$\mu_k,\Sigma_k,\pi_k$を再計算する。

$$\mu_k^{new} = \frac{1}{N_k}\sum_{n=1}^{N}\gamma(z_{nk})\vec{x}_n$$

$$\Sigma_k^{new} = \frac{1}{N_k}\sum_{n=1}^{N}\gamma(z_{nk})(\vec{x}_n - \mu_k^{new})(x_n - \mu_k^{new})^T$$

$$\pi_k^{new} = \frac{N_k}{N}, \quad N_k = \sum_{n=1}^{N}\gamma(z_{nk})$$

4. $\mu_k^{new},\Sigma_k^{new},\pi_k^{new}$で対数尤度()を計算。対数尤度、もしくはパラメータの値の変化を見て収束性を確認。収束していなければ、
2に戻る。

なお、上記の更新ステップでは対数尤度関数は必ず増加することが保証されている。(Column)

