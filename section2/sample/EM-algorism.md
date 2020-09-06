# 講習会資料Sample

## $Column\sim$ 対数尤度関数の最大化とKL-divergence $\sim$

EMアルゴリズムで対数尤度を最大化する際、各ステップで対数尤度関数が増加(もしくは不変)する
ことをEMアルゴリズムを一般的に取り扱うことで示すことができる。パラメータを
まとめて$\theta$として表すと、最大化する対数尤度関数は、

$$\ln p(X | \theta) = \ln \sum_Z p(X,Z|\theta)$$
$$\qquad \qquad = \ln\sum_{Z}q(Z)\frac{p(X,Z|\theta)}{q(Z)}$$
$$\qquad \qquad  \geq \sum_{Z} q(Z)\ln \frac{p(X,Z|\theta)}{q(Z)}$$

ここで$q(Z)$は潜在変数$Z$についての分布である。なお、3行目の変形ではイェンセン不等式、

$$\ln E\ [x] \geq E\ [\ln x]$$

を用いた。最終的に求められた式、

$$L(q,\theta) = \sum_{Z} q(Z)\ln \frac{p(X,Z|\theta)}{q(Z)}$$

には変分下界という名前がつけられている。($L(q,\theta)$は$\theta$の関数、$q(Z)$の汎関数となっている。汎関数については次コラム参照。)　次に対数尤度関数$\ln p(X|\theta)$とこの変分下界$L(q,\theta)$の差について考えると、

$$\ln p(X|\theta) - L(q,\theta)\qquad \qquad$$
$$= \ ......\qquad \qquad$$
$$= \ ......\qquad \qquad$$
$$\qquad \qquad= - \sum_{Z}q(Z)\ln\frac{p(Z|X,\theta)}{q(Z)}$$
$$\qquad =KL[q(Z)||p(Z|X,\theta)]$$

となる。よって対数尤度関数は、

$$\ln p(X|\theta) = L(q,\theta) + KL[q(Z)||p(Z|X,\theta)]$$

と変形することができることがわかる。

この対数尤度関数の最尤解を求めるためには以下のような方法をとると、常に
$\ln p(X|\theta)$が増加するように最適化を行うことができる。

1. $L(q,\theta)$を$q(Z)$について最大化する。(Eステップ)
2. $L(q,\theta)$を$\theta$について最大化する。(Mステップ)
3. 1に戻る

現在のパラメータ$\theta$の値を$\theta = \theta^{old}$としてそれぞれのステップについて詳しく見ていく。

まず1のステップ(Eステップ)では$q(Z)$についての最大化を行う。その際、左辺の$\ln p(X|\theta)$は$q(Z)$に依存しないので、このステップにおいて不変である。このこと
から$L(q,\theta)$の最大化は、$KL[q(Z)||p(Z|X,\theta)]$の最小化と等価
になり、$L(q,\theta)$の最大値はKLダイバージェンスが0になるとき、つまり、

$$q(Z) = p(Z|X,\theta^{old})$$

であることがわかる。またこのとき当然ながら変分下界は対数尤度と一致する。(図)

次に2のステップ(Mステップ)について考える。1のステップが終わった段階で、
$KL[q(Z)||p(Z|X,\theta)]$、$L(q,\theta)$の値は、

$$KL[q(Z)||p(Z|X,\theta)] = 0$$
$$L(q,\theta) = \sum_{Z} q(Z)\ln \frac{p(X,Z|\theta)}{q(Z)} \qquad \qquad$$
$$\qquad \quad= \sum_{Z} p(Z|X,\theta^{old})\ln \frac{p(X,Z|\theta)}{p(Z|X,\theta^{old})}$$
$$= \sum_{Z} p(Z|X,\theta^{old})\ln p(X,Z|\theta) - \sum_{Z} p(Z|X,\theta^{old})\ln p(Z|X,\theta^{old})$$
$$= Q(\theta, \theta^{old}) + \rm const. \quad \qquad$$

となる。なお、第2項は$\theta$には依存しないので定数扱いしている。
よって$L(q,\theta)$を最大化するには、$Q(\theta, \theta^{old})$を最大化
すればよい。また、KLダイバージェンスについても、KLダイバージェンス自体が正の
値しか取り得ないことと、Eステップが終わった時点で0となっていることを考えれば、Mステップにおいてどんな$\theta=\theta^{new}$をとったところで必ず
増加することがわかる。(図)

以上より、このような方法で対数尤度$\ln p(X|\theta)$を最大化すれば、パラメータ更新手続きは常に対数尤度を増加させることがわかる。(この更新手続きを
視覚化したのが図である。)

#### 補足
　Mステップにおいて、$L(q,\theta)$を最大化させるのに$Q(\theta, \theta^{old})$を最大化させたが、これを実際に行う際には、そもそも$Q(\theta, \theta^{old})$が最適化可能であるという前提が必要となる。$Q(\theta, \theta^{old})$に現れる同時分布$p(X,Z|\theta)$の尤度は、「完全データ対数尤度関数」と呼ばれており、上記の式変形は、これが十分簡単に最大化することができること
を仮定したものである。(例えば、$p(X,Z|\theta)$が指数型分布族である場合は、対数と指数がキャンセルして$\ln p(X,Z|\theta)$の最大化は用意になる。)