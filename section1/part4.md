## 1-5 SVM（サポートベクトルマシン）

### 1-5-1 SVMの主問題

次のようなクラス分類の問題を考える。

$$t_i(w^Tx_i + b) > 0, \quad t_i = 1,2,...n, \quad t_{i} =
        \begin{cases}
            1 & x_i \in K_1 \\
            -1 & x_i \in K_2 \\
        \end{cases} $$

この問題を解くために、p次元データ$x = (x_1,x_2,..., x_p)$と超平面$w^Tx_i + b = 0$の距離を

$$d = \frac{|w_1x_1+w_2x_2+w_3x_3+b|}{\sqrt{w_1^2+w_2^2+...w_p^2}} = \frac{|w^Tx_i + b|}{||w||}$$


とし、2つのクラスを分ける超平面とそれに最も近いデータ(サポートベクトル)との
間の距離(マージン$M$)を最大化するように$w,b$を最適化する。

$$argmax_{w,b} M,\quad \frac{t_i(w^Tx_i + b)}{||w||} \geq M,\quad i=1,2,...n$$

これに簡単な式変形を加えると次のようにこの最適化問題を書き換えることができる。

$$ argmin_{w,b}\frac{1}{2}||w||^2,\quad t_i(w^Tx_i+b)\geq 1, \quad i=1,2,...n$$

また、線形分離可能でない場合にスラッグ変数$\epsilon_i$を導入してこの最適化問題の制約を緩めることができる。

$$argmin_{w,b}\Bigl\{\frac{1}{2}||w||^2 + C\sum_{i=0}^{n}\epsilon_i\Bigl\},\quad t_i(w^Tx_i+b)\geq 1-\epsilon_i,\quad \epsilon_i \geq 0 \quad i=1,2,...n$$

これはソフトマージンSVMと呼ばれている。またこれらの最適化問題は、不等式制約条件最適化問題の「主問題」と
呼ばれており、主問題に対して「双対問題」という形式の問題を導くことができる。

(1)式()から式()への式変形をせよ。

### 1-5-2 SVMの双対問題と非線形分離

ラグランジュ未定乗数法を用いると、主問題を双対問題に変形することができる。

$$argmax_{\alpha}\Biggl\{L(\alpha) = \sum_{i=1}^{n}\alpha_i - \frac{1}{2}\sum_{i=1}^{n}\sum_{i=j}^{n}\alpha_i\alpha_iy_iy_ix_i^Tx_j\Biggl\}$$
$$\sum_{i=1}^{n}\alpha_iy_i= 0 ,\quad 0 \leq \alpha_i \leq C, \quad i = 1,2,...n$$

学習データが線形識別関数で分離できない場合は、高次元非線形空間にデータ点を
写像し、その空間内で線形識別関数を用いると、線形分離可能となる可能性がある。主問題を双対問題に変形しておくと、学習データの高次元非線形空間への写像は単に

$$argmax_{\alpha}\Biggl\{L(\alpha) = \sum_{i=1}^{n}\alpha_i - \frac{1}{2}\sum_{i=1}^{n}\sum_{i=j}^{n}\alpha_i\alpha_iy_iy_i\Phi(x)_i^T\Phi(x)_j\Biggl\}$$
$$\sum_{i=1}^{n}\alpha_iy_i= 0 ,\quad 0 \leq \alpha_i \leq C, \quad i = 1,2,...n$$

と問題を変えるだけでよいので定式化が容易であり、さらにカーネル法(次節参照)を適用できる形式になる。

(2)式()の最適化問題にラグランジュ未定乗数法を適用することで得られる以下の
ラグランジュ関数

$$L(w,b,\epsilon,\alpha,\beta) = \frac{1}{2}||w||^2 + C\sum_{i=1}^{n}\epsilon_i - \sum_{i=1}^{n}\alpha_i\{t_i(w^Tx_i + b)-1 + \epsilon_i\}-\sum_{i=1}^{n}\beta_i\epsilon_i$$

　について、主変数$w,b,\epsilon$の偏微分を考え、双対問題()を導け。

### 1-5-3 カーネル法

双対問題に現れる値$\Phi(x)_i$は計算量が多く最適化問題を解くことが困難な形に
なってしまっているが、$\Phi(x)_i$の内積$\Phi(x)_i^T\Phi(x)_j$は
$\Phi(x)_i$を計算しないで求めることができることが知られている。

$$K(x_i, x_j) = \Phi(x)_i^T\Phi(x)_j$$

$K(x_i, x_j)$をカーネル関数と呼び、このようにして内積を計算する手法を「カーネル法」とも呼ぶ。(これを決めてしまえば内積計算$\Phi(x)_i^T\Phi(x)_j$を簡単に行うことができてしまうため、この一見魔法のよう
な手法は「カーネルトリック」も呼ばれている。)

以下にいくつかのカーネル関数をあげる。(実際は、解きたい問題に合わせてこれらを
使い分ける。)

ガウスカーネル

$$K(x_i, x_j) = \exp\Biggl\{-\frac{||x_i - x_j||^2}{2\sigma^2}\Biggl\}$$

多項式カーネル

$$K(x_i, x_j) = (x_i^Tx_j + c)^d$$

シグモイドカーネル

$$K(x_i, x_j) = tanh(bx_i^Tx_j + c)$$
