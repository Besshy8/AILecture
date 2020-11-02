## 1-5 SVM（サポートベクトルマシン）

### 1-5-1 SVMの主問題

次のようなクラス分類の問題を考える。

$$t_i(\vec{w}^T\vec{x}_i + b) > 0, \quad t_i = 1,2,...n, \quad t_{i} =
        \begin{cases}
            1 & x_i \in K_1 \\
            -1 & x_i \in K_2 \\
        \end{cases} \tag{1.5.1}$$

この問題を解くために、p次元データ$\vec{x_i} = (x_1,x_2,..., x_p)$と超平面$\vec{w}^T\vec{x}_i + b = 0$の距離を

$$d = \frac{|w_1x_1+w_2x_2+\cdots + w_px_p+b|}{\sqrt{w_1^2+w_2^2+...w_p^2}} = \frac{|\vec{w}^T\vec{x}_i + b|}{||\vec{w}||} \tag{1.5.2}$$


とし、2つのクラスを分ける超平面とそれに最も近いデータ(サポートベクトル)との
間の距離(マージン$M$)を最大化するように$w,b$を最適化する。

$$argmax_{\vec{w},b} M,\quad \frac{t_i(\vec{w}^T\vec{x}_i + b)}{||\vec{w}||} \geq M,\quad i=1,2,...n\tag{1.5.3}$$

これに簡単な式変形を加えると次のようにこの最適化問題を書き換えることができる。

$$ argmin_{\vec{w},b}\frac{1}{2}||\vec{w}||^2,\quad t_i(\vec{w}^T\vec{x}_i+b)\geq 1, \quad i=1,2,...n\tag{1.5.4}$$

また、線形分離可能でない場合にスラッグ変数$\epsilon_i$を導入してこの最適化問題の制約を緩めることができる。

$$argmin_{\vec{w},b}\Bigl\{\frac{1}{2}||\vec{w}||^2 + C\sum_{i=0}^{n}\epsilon_i\Bigl\},\quad t_i(\vec{w}^T\vec{x}_i+b)\geq 1-\epsilon_i,\quad \epsilon_i \geq 0 \quad i=1,2,...n \tag{1.5.5}$$

これはソフトマージンSVMと呼ばれている。またこれらの最適化問題は、不等式制約条件最適化問題の「主問題」と
呼ばれており、主問題に対して「双対問題」という形式の問題を導くことができる。

(1)式(1.5.3)から式(1.5.4)への式変形をせよ。

\clearpage

### 1-5-2 SVMの双対問題と非線形分離

ラグランジュ未定乗数法を用いると、上記の主問題を双対問題に変形することができる。

$$argmax_{\alpha}\Biggl\{L(\alpha) = \sum_{i=1}^{n}\alpha_i - \frac{1}{2}\sum_{i=1}^{n}\sum_{j=1}^{n}\alpha_i\alpha_jt_it_i\vec{x}_i^T\vec{x}_j\Biggl\} \tag{1.5.6}$$
$$\sum_{i=1}^{n}\alpha_it_i= 0 ,\quad 0 \leq \alpha_i \leq C, \quad i = 1,2,...n$$

学習データが線形識別関数で分離できない場合は、高次元非線形空間にデータ点を
写像し、その空間内で線形識別関数を用いると、線形分離可能となる可能性がある。主問題を双対問題に変形しておくと、学習データの高次元非線形空間への写像は単に

$$argmax_{\alpha}\Biggl\{L(\alpha) = \sum_{i=1}^{n}\alpha_i - \frac{1}{2}\sum_{i=1}^{n}\sum_{j=1}^{n}\alpha_i\alpha_jt_it_j\Phi(\vec{x})_i^T\Phi(\vec{x})_j\Biggl\} \tag{1.5.7}$$
$$\sum_{i=1}^{n}\alpha_it_i= 0 ,\quad 0 \leq \alpha_i \leq C, \quad i = 1,2,...n$$

と問題を変えるだけでよいので定式化が容易であり、さらにカーネル法(次節参照)を適用できる形式になる。

(2)式(1.5.5)の最適化問題にラグランジュ未定乗数法を適用することで得られる以下の
ラグランジュ関数

$$L(\vec{w},b,\epsilon,\alpha,\beta) = \frac{1}{2}||\vec{w}||^2 + C\sum_{i=1}^{n}\epsilon_i - \sum_{i=1}^{n}\alpha_i\{t_i(\vec{w}^T\vec{x}_i + b)-1 + \epsilon_i\}-\sum_{i=1}^{n}\beta_i\epsilon_i \tag{1.5.8}$$

　について、主変数$w,b,\epsilon$の偏微分を考え、双対問題(1.5.6)を導け。

\clearpage

### 1-5-3 カーネル法

双対問題に現れる値$\Phi(\vec{x})_i$は計算量が多く最適化問題を解くことが困難な形に
なってしまっているが、$\Phi(\vec{x})_i$の内積$\Phi(\vec{x})_i^T\Phi(\vec{x})_j$は
$\Phi(\vec{x})_i$を計算しないで求めることができることが知られている。

$$K(\vec{x}_i, \vec{x}_j) = \Phi(\vec{x})_i^T\Phi(\vec{x})_j \tag{1.5.9}$$

$K(\vec{x}_i, \vec{x}_j)$をカーネル関数と呼び、このようにして内積を計算する手法を「カーネル法」とも呼ぶ。(これを決めてしまえば内積計算$\Phi(\vec{x})_i^T\Phi(\vec{x})_j$を簡単に行うことができてしまうため、この一見魔法のよう
な手法は「カーネルトリック」も呼ばれている。)

以下にいくつかのカーネル関数をあげる。(実際は、解きたい問題に合わせてこれらを
使い分ける。)

ガウスカーネル

$$K(\vec{x}_i, \vec{x}_j) = \exp\Biggl\{-\frac{||\vec{x}_i - \vec{x}_j||^2}{2\sigma^2}\Biggl\} \tag{1.5.10}$$

多項式カーネル

$$K(\vec{x}_i, \vec{x}_j) = (\vec{x}_i^T\vec{x}_j + c)^d \tag{1.5.11}$$

シグモイドカーネル

$$K(\vec{x}_i, \vec{x}_j) = tanh(b\vec{x}_i^T\vec{x}_j + c) \tag{1.5.12}$$

(3) 


上記の問題(3)のように、例えば多項式カーネルでは、$d$次以下の全種類の単項式を各成分に持つような特徴ベクトル$\Phi(x)$の内積を求めていることに対応します。(カーネルを決めたら自動的に写像のされ方が決定するので、問題に合わせてカーネルを選定する必要が出てきます。)
