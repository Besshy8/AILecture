## $Column\sim$ 指数型分布族  $\sim$
　本章で取り上げる尤度関数、共役事前分布には一般形が存在する。まず、
尤度関数については一般的に次のような形のものを想定すると都合がいい。
この形で表される分布の族のことを、「指数型分布族」と呼ぶ。

$$p(\vec{x}|\vec{\eta})=h(\vec{x})\exp(\vec{\eta}^{T}\vec{t}(\vec{x})-a(\vec\eta)) \tag{C.2.2.1}$$

$\eta$は自然パラメータ、$\vec{t}(\vec{x}_n)$は十分統計量、$h(\vec{x})$は基底測度、$a(\vec{\eta})$は対数分配関数と呼ばれている。
対数分配関数$a(\vec{\eta})$は$p(\vec{x}|\vec{\eta})$を積分して
$1$になるように保証してくれるもので、

$$\int h(\vec{x})\exp(\vec{\eta}^{T}\vec{t}(\vec{x})-a(\vec\eta)) \ dx = 1 \tag{C.2.2.2}$$

$$a(\vec{\eta})=ln\int h(\vec{x})\exp(\vec{\eta}^{T}\vec{t}(\vec{x})) \ dx \tag{C.2.2.3}$$

この指数型分布族に対して都合のいい共役事前分布は次のようなものが知られている。

$$p_{\lambda}(\vec{\eta})=h(\vec{\eta})\exp(\vec{\eta}^{T}\vec{\lambda_1}-a(\vec\eta)\lambda_2-a_c(\vec{\lambda})) \tag{C.2.2.4}$$

この共役事前分布を用いて事後分布を計算すると、

$$p(\vec{\eta}|X) \propto p_\lambda(\vec{\eta})\prod_{n=1}^{N}p(\vec{x}_n|\vec{\eta}) \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad$$
$$=h(\vec{\eta})\exp(\vec{\eta}^{T}\vec{\lambda_1}-a(\vec\eta)\lambda_2-a_c(\vec{\lambda}))\prod_{n=1}^Nh(\vec{x})\exp(\vec{\eta}^{T}\vec{t}(\vec{x})-a(\vec\eta)) \qquad \qquad \qquad$$
$$=h(\vec{\eta})\exp(\vec{\eta}^{T}\vec{\lambda_1}-a(\vec\eta)\lambda_2-a_c(\vec{\lambda})) \ \biggl\{\prod_{n=1}^Nh(\vec{x})\biggl\} \ \exp\biggl(\vec{\eta}^{T}\sum_{n=1}^N(\vec{t}(\vec{x})-Na(\vec\eta))\biggl)$$
$$\propto h(\vec{\eta})\exp\biggl(\vec{\eta}^{T}\biggl(\vec{\lambda_1}+\sum_{n=1}^{N}\vec{t}(\vec{x}_n)\biggl)-a(\vec\eta)(\lambda_2+N)\biggl) \qquad \qquad \qquad \qquad \tag{C.2.2.5}$$

このように事後分布も事前分布と同じ形になる。事後分布のパラメータは、

$$\vec{\lambda_1}=\vec{\lambda_1}+\sum_{n=1}^{N}\vec{t}(\vec{x}_n), \quad \lambda_2=\lambda_2+N \tag{C.2.2.6}$$

予測分布についても同じように指数分布族で表すことができ、

$$p(\vec{x}_*|X)=\int p(\vec{x}_*|\vec{\eta})p(\vec{\eta}|X)d\vec{\eta} \qquad \qquad \qquad \qquad \qquad \qquad \qquad \qquad$$
$$= \int d\vec{\eta} \ \bigl\{h(\vec{x})\exp(\vec{\eta}^{T}\vec{t}(\vec{x})-a(\vec\eta))h_c(\vec{\eta})\exp(\vec{\eta}^{T}\vec{\lambda_1}-a(\vec\eta)\lambda_2-a_c(\vec{\lambda}))\bigl\} $$ 
$$ = \cdots \cdots \cdots= h(\vec{x}_*)\frac{\exp(a_c(\vec{\lambda_1}+\vec{t}(\vec{x_*}),\lambda_2+1))}{\exp(a_c(\vec{\lambda_1},\lambda_2))} \qquad \qquad \qquad \qquad \qquad \tag{C.2.2.7}$$

となり、一般的には指数型分布族にはならないことがわかる。ベルヌーイ分布、
ガウス分布など、多くの分布が指数型分布族として表せることが知られており、
この形で表すことができれば、あとは式(C.2.2.5)や(C.2.2.7)に代入するだけで事後分布、
予測分布を計算することができる。

　また、計算の都合上、$g(\vec{\eta})=\exp(-a(\vec{\eta}))$と置き、(C.2.2.2)の両辺に関して$\vec{\eta}$について勾配を取ると、

$$\nabla_{\vec{\eta}}\int h(\vec{x})\exp(\vec{\eta}^{T}\vec{t}(\vec{x})-a(\vec\eta)) \ dx = 0 \tag{C.2.2.8}$$

$$\nabla_{\vec{\eta}}g(\vec{\eta})\int h(\vec{x})\exp(\vec{\eta}^{T}\vec{t}(\vec{x})) \ dx $$
$$\qquad \qquad \qquad +\int d\vec{x} \ \bigl\{h(\vec{x})\exp(\vec{\eta}^{T}\vec{t}(\vec{x}))\vec{t}(\vec{x})\bigl\} = 0$$

$$-\frac{1}{g(\vec{\eta})}\nabla_{\vec{\eta}}=g(\vec{\eta})\int d\vec{x} \ \bigl\{h(\vec{x})\exp(\vec{\eta}^{T}\vec{t}(\vec{x}))\vec{t}(\vec{x})\bigl\} \ =E[\vec{t}(\vec{x})] \tag{C.2.2.9}$$

よって次の結果が得られる。

$$-\nabla_{\eta}ln \ g(\vec{\eta})=\nabla_{\vec{\eta}}a(\vec{\eta})=E[\vec{t}(\vec{x})] \tag{C.2.2.10}$$

このことから、対数分配関数$a(\vec{\eta})$の$\vec{\eta}$に関する勾配
は十分統計量$\vec{t}(\vec{x})$の期待値になる。

また2階の偏微分は十分統計量の共分散になることもわかる。

$$\frac{\partial^2a(\vec{\eta})}{\partial\eta_i\partial\eta_j}=E[\vec{t}_i(\vec{x})\vec{t}_j(\vec{x})]-E[\vec{t}_i(\vec{x})]E[\vec{t}_j(\vec{x})]$$

$$=Cov[\vec{t}_i(\vec{x}),\vec{t}_j(\vec{x})] \tag{C.2.2.11}$$

