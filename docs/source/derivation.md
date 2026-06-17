---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.19.3
kernelspec:
  name: python3
  language: python
  display_name: python3
---

# Derivation

This page derives several equations that appear in {ref}`sec:gates` and {ref}`sec:theory`.

+++

## Preparation

### Variables and parameters

```{code-cell}
from itertools import product
from math import isclose

import numpy as np
import sympy
from sympy import (
    Eq,
    Matrix,
    acos,
    atan,
    cos,
    cot,
    exp,
    expand_trig,
    pi,
    simplify,
    sin,
    sqrt,
    symbols,
    tan,
    trigsimp,
)

sympy.init_printing()

x1, p1, x2, p2 = symbols("x1,p1,x2,p2")
c, r, t, t1, t2, phi = symbols(r"c,r,θ,θ1,θ2,\phi")
g, kappa, eta, x = symbols(r"g,\kappa,\eta,x")
a, b = symbols("a,b")
alpha, beta, lam = symbols(r"\alpha,\beta,\lambda")
largeR, theta_rel = symbols(r"R,\theta_\text{rel}")
ta, tb, tc, td = symbols("θ_a,θ_b,θ_c,θ_d")
m1, m2 = symbols("m1,m2")
ma, mb, mc, md = symbols("m_a,m_b,m_c,m_d")
mak, mbk, mck, mdk = symbols("m_a^k,m_b^k,m_c^k,m_d^k")
tak, tbk, tck, tdk = symbols("θ_a^k,θ_b^k,θ_c^k,θ_d^k")
makm1, mbkm1, mckm1, mdkm1 = symbols("m_a^{k-1},m_b^{k-1},m_c^{k-1},m_d^{k-1}")
takm1, tbkm1, tckm1, tdkm1 = symbols("θ_a^{k-1},θ_b^{k-1},θ_c^{k-1},θ_d^{k-1}")
makmN, mbkmN, mckmN, mdkmN = symbols("m_a^{k-N},m_b^{k-N},m_c^{k-N},m_d^{k-N}")
takmN, tbkmN, tckmN, tdkmN = symbols("θ_a^{k-N},θ_b^{k-N},θ_c^{k-N},θ_d^{k-N}")
tak1, tbk1, tck1, tdk1 = symbols("θ_a^{k+1},θ_b^{k+1},θ_c^{k+1},θ_d^{k+1}")
takN, tbkN, tckN, tdkN = symbols("θ_a^{k+N},θ_b^{k+N},θ_c^{k+N},θ_d^{k+N}")
xak1, xbk1, xck1, xdk1 = symbols("x_a^{k+1},x_b^{k+1},x_c^{k+1},x_d^{k+1}")
pak1, pbk1, pck1, pdk1 = symbols("p_a^{k+1},p_b^{k+1},p_c^{k+1},p_d^{k+1}")
xakN, xbkN, xckN, xdkN = symbols("x_a^{k+N},x_b^{k+N},x_c^{k+N},x_d^{k+N}")
pakN, pbkN, pckN, pdkN = symbols("p_a^{k+N},p_b^{k+N},p_c^{k+N},p_d^{k+N}")
```

### Operations

+++

#### Phase rotation

```{code-cell}
def R(phi):
    return Matrix([
        [cos(phi), -sin(phi)],
        [sin(phi), cos(phi)],
    ])


R(phi)
```

#### Squeezing

```{code-cell}
def S(r):
    return Matrix([
        [exp(-r), 0],
        [0, exp(r)],
    ])


def Sv(c):
    return Matrix([
        [1 / c, 0],
        [0, c],
    ])


S(r), Sv(c)
```

#### Shear (X-invariant)

```{code-cell}
def P(k):
    return Matrix([
        [1, 0],
        [2 * k, 1],
    ])


P(kappa)
```

#### Shear (P-invariant)

```{code-cell}
def Q(eta):
    return Matrix([
        [1, 2 * eta],
        [0, 1],
    ])


Q(eta)
```

#### Controlled-Z

```{code-cell}
def Cz(g):
    return Matrix([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, g, 1, 0],
        [g, 0, 0, 1],
    ])


Cz(g)
```

#### Beamsplitter interaction

```{code-cell}
def B_ab(alpha, beta):
    cos_p = cos(alpha + beta)
    cos_m = cos(alpha - beta)
    sin_p = sin(alpha + beta)
    sin_m = sin(alpha - beta)
    return Matrix([
        [cos_p * cos_m, sin_p * sin_m, -sin_p * cos_m, sin_m * cos_p],
        [sin_p * sin_m, cos_p * cos_m, sin_m * cos_p, -sin_p * cos_m],
        [sin_p * cos_m, -sin_m * cos_p, cos_p * cos_m, sin_p * sin_m],
        [-sin_m * cos_p, sin_p * cos_m, sin_p * sin_m, cos_p * cos_m],
    ])


B_ab(alpha, beta)
```

```{code-cell}
def B(sqrt_r, theta_rel):
    h_theta = theta_rel / 2
    h_acos = acos(sqrt_r) / 2
    alpha = h_theta + h_acos
    beta = h_theta - h_acos
    return B_ab(alpha, beta)


B(sqrt(largeR), theta_rel)
```

#### Two-mode shear

```{code-cell}
def P2(a, b):
    return Matrix([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [2 * a, b, 1, 0],
        [b, 2 * a, 0, 1],
    ])


P2(a, b)
```

#### 50:50 beamsplitter and inverse 50:50 beamsplitter

```{code-cell}
Bd = Matrix([
    [1, -1, 0, 0],
    [1, 1, 0, 0],
    [0, 0, 1, -1],
    [0, 0, 1, 1],
]) / sqrt(2)
Bu = Matrix([
    [1, 1, 0, 0],
    [-1, 1, 0, 0],
    [0, 0, 1, 1],
    [0, 0, -1, 1],
]) / sqrt(2)
Bd, Bu
```

#### Foursplitter

```{code-cell}
A_block = (
    Matrix([
        [1, 1, -1, -1],
        [-1, 1, 1, -1],
        [1, 1, 1, 1],
        [-1, 1, -1, 1],
    ])
    / 2
)
Ad = Matrix.diag(A_block, A_block)
simplify(Ad)
```

#### {math}`V` in Eq. [](#eq:teleportation-circuit)

```{code-cell}
def V(t1, t2):
    return R((t1 + t2 - pi) / 2) * Sv(cot((t1 - t2) / 2)) * R((t1 + t2) / 2)
```

## Gates

In this section, we confirm that the measurement angles shown in {ref}`sec:gates-in-graph-repr` implement the desired operations.

### Single mode gates

As stated in {ref}`sec:through-and-swap`, in the case of a single-mode operation, the same operation is applied to both input modes within each macronode.
Therefore, it is sufficient to show that substituting {math}`\theta_a` and {math}`\theta_b` into {math}`V\left(\theta_b, \theta_a\right)` implements the desired operation, where {math}`V` is defined in Eq. [](#eq:teleportation-circuit).

#### `Wiring`

Validate the measurement angles in {ref}`sec:gate-graph-repr-wiring`.

```{code-cell}
V(pi / 2, 0)
```

This is identity operation.

+++

#### `PhaseRotation`

Validate the measurement angles in {ref}`sec:gate-graph-repr-rotation`.

```{code-cell}
source = simplify(V((phi + pi) / 2, phi / 2))
assert Eq(source, R(phi))
```

#### `Squeezing`

Validate the measurement angles in {ref}`sec:gate-graph-repr-squeezing`.

```{code-cell}
source = simplify(V(t, -t))
target = simplify(R(-pi / 2) * Sv(cot(t)))
assert Eq(source, target)
```

#### `Squeezing45`

Validate the measurement angles in {ref}`sec:gate-graph-repr-squeezing45`.
Multiply both sides by {math}`R\left(\frac{\pi}{4}\right)` from the left and by {math}`R\left(-\frac{\pi}{4}\right)` from the right.

```{code-cell}
source = simplify(R(pi / 4) * V(pi / 4 + t, pi / 4 - t) * R(-pi / 4))
target = simplify(Sv(cot(t)))
assert Eq(source, target)
```

(sec:shear-x-invariant-validation)=

#### `ShearXInvariant`

Validate the measurement angles in {ref}`sec:gate-graph-repr-shear-x-inv`.

```{code-cell}
source = V(pi / 2, ta)
source = simplify(source)
source = expand_trig(source)
source = trigsimp(source)
source = simplify(source)
source = source.subs(ta, atan(kappa))
assert Eq(source, P(kappa))
```

#### `ShearPInvariant`

Validate the measurement angles in {ref}`sec:gate-graph-repr-shear-p-inv`.

```{code-cell}
source = V(tb, 0)
source = simplify(source)
source = expand_trig(source)
source = source.subs(tan(tb / 2), sqrt((1 - cos(tb)) / (1 + cos(tb))))  # half-angle formula
source = source.subs(tb, sympy.acot(eta))
source = simplify(source)
source
```

It can be easily shown that the (0, 1) element of matrix `source` is {math}`2\eta`, which implements the desired operation.

```{code-cell}
assert Eq(source.subs(source[0, 1], 2 * eta), Q(eta))
```

(sec:graph-arbitrary-angles-validation)=

#### `Arbitrary`

Solving Eqs. [](#eq:general-1mode-condition) yields the following measurement angles.

```{math}
\begin{aligned}
\theta_a^k & =\beta-\arctan e^{-\lambda} \\
\theta_b^k & =\beta+\arctan e^{-\lambda} \\
\theta_a^{k+N} & =\frac{\alpha-\beta}{2}+\frac{\pi}{4} \\
\theta_b^{k+N} & =\frac{\alpha-\beta}{2}+\frac{3 \pi}{4}
\end{aligned}
```

As stated in {ref}`sec:angle-equivalence-2pi-pi`, shifting the measurement angles by {math}`\pi` does not affect the operation, feedforward, or displacement.
Therefore, prioritizing the symmetry of the equation, we rewrite {math}`\theta_b^{k+N}` as

```{math}
\theta_b^{k+N} =\frac{\alpha-\beta}{2}-\frac{\pi}{4}.
```

These measurement angles are the same as those shown in {ref}`sec:gate-graph-repr-arbitrary`.
Validate the angles.

```{code-cell}
source = V((alpha - beta) / 2 - pi / 4, (alpha - beta) / 2 + pi / 4) * V(
    beta + atan(exp(-lam)), beta - atan(exp(-lam))
)
source = simplify(source)
target = R(alpha) * S(lam) * R(beta)
assert Eq(source, target)
```

### Two mode gates

As seen in {numref}`fig:macronode_circuit`, the desired operation {math}`U` consists of independently applying the operation {math}`V\left(\theta_b, \theta_a\right)` to mode {math}`b` and {math}`V\left(\theta_d, \theta_c\right)` to mode {math}`d`, with a beam splitter and an inverse beam splitter on either side. Let

```{math}
V_2\left(\theta_a, \theta_b, \theta_c, \theta_d\right)=\left(\begin{array}{cccc}
V_b^{00} & & V_b^{01} & \\
& V_d^{00} && V_d^{01} \\
V_b^{10} && V_b^{11} & \\
& V_d^{10} && V_d^{11}
\end{array}\right)
```

represents the operation that applies

```{math}
\begin{aligned}
& V\left(\theta_b, \theta_a\right)=\left(\begin{array}{ll}
V_b^{00} & V_b^{01} \\
V_b^{10} & V_b^{11}
\end{array}\right), \\
& V\left(\theta_d, \theta_c\right)=\left(\begin{array}{ll}
V_d^{00} & V_d^{01} \\
V_d^{10} & V_d^{11}
\end{array}\right)
\end{aligned}
```

in parallel to the two modes. Then, the desired operation can be written as
```{math}
U\left(\theta_a, \theta_b, \theta_c, \theta_d\right)
=
B_\uparrow V_2\left(\theta_a, \theta_b, \theta_c, \theta_d\right) B_\downarrow.
```

+++

#### `BeamSplitter`

Validate the measurement angles in {ref}`sec:gate-graph-repr-bs`.
The angles can be expressed as {math}`\left(\alpha, \alpha + \pi / 2, \beta, \beta + \pi / 2\right)` using {math}`\alpha` and {math}`\beta` from Eq. [](#eq:bs-alpha-beta).

```{code-cell}
Vb = simplify(V(alpha + pi / 2, alpha))
Vd = simplify(V(beta + pi / 2, beta))
V2 = Matrix([
    [Vb[0, 0], 0, Vb[0, 1], 0],
    [0, Vd[0, 0], 0, Vd[0, 1]],
    [Vb[1, 0], 0, Vb[1, 1], 0],
    [0, Vd[1, 0], 0, Vd[1, 1]],
])
source = trigsimp(Bu * V2 * Bd)

target = B_ab(alpha, beta)
diff = simplify(source - target)
diff
```

#### `ControlledZ`

Calculate {math}`B_\downarrow C_Z(g) B_\uparrow` to get {math}`V_2`.

```{code-cell}
V2 = Bd * Cz(g) * Bu
V2
```

Referring to {ref}`sec:shear-x-invariant-validation`, it can be seen that {math}`V\left(\theta_b, \theta_a\right)=P(-\frac{g}{2})` and {math}`V\left(\theta_d, \theta_c\right)=P(\frac{g}{2})`.
These measurement angles are exactly the same as the measurement angles in {ref}`sec:gate-graph-repr-cz`.

+++

#### `TwoModeShear`

Validate the measurement angles in {ref}`sec:gate-graph-repr-2mode-shear`.

```{code-cell}
Vb = simplify(V(pi / 2, ta))
Vd = simplify(V(pi / 2, tc))
V2 = Matrix([
    [Vb[0, 0], 0, Vb[0, 1], 0],
    [0, Vd[0, 0], 0, Vd[0, 1]],
    [Vb[1, 0], 0, Vb[1, 1], 0],
    [0, Vd[1, 0], 0, Vd[1, 1]],
])
source = simplify(Bu * V2 * Bd)

# Verify that Bu * V2 * Bd equals P2(a, b) for various combinations of a and b, using float comparison
parameters = np.linspace(-3, 3, 7)
idxs = list(range(4))
for a, b in product(parameters, parameters):
    source = source.subs([(ta, atan(a - b / 2)), (tc, atan(a + b / 2))])
    source_f = source.evalf()
    P2_f = P2(a, b).evalf()
    for i, j in product(idxs, idxs):
        isclose(source_f[i, j], P2_f[i, j], abs_tol=1e-9)
```

(sec:std-bs-derivation)=

## `mqc3.circuit.ops.std.BeamSplitter`

In this section, it is shown that the operation matrix in {ref}`sec:gate-circuit-std-bs` can be decomposed as

```{math}
---
label: eq:std-bs-derivation-decomposition
---
R_{12} \left(\pi-\theta-\phi, \frac{\pi}{2}-\theta\right) Man\left(0, \frac{\pi}{2}, \theta, \theta+\frac{\pi}{2}\right) R_{12} \left(\phi-\pi, -\frac{\pi}{2}\right).
```

First, define {math}`R_{12}`.

```{code-cell}
def R12(t1, t2):
    return Matrix([
        [cos(t1), 0, -sin(t1), 0],
        [0, cos(t2), 0, -sin(t2)],
        [sin(t1), 0, cos(t1), 0],
        [0, sin(t2), 0, cos(t2)],
    ])


R12(t1, t2)
```

The target matrix can be expressed as

```{math}
R_{12} \left(\pi-\theta-\phi, \frac{\pi}{2}-\theta\right) B^{\prime}\left(\alpha=0, \beta=\theta\right) R_{12} \left(\phi-\pi, -\frac{\pi}{2}\right)
```

using the right-hand side of the definition of {ref}`sec:gate-bs-symbol`,
```{math}
B^{\prime}\left(\alpha, \beta\right)=
\left(\begin{array}{rrrr}
\cos (\alpha+\beta) \cos (\alpha-\beta) & \sin (\alpha+\beta) \sin (\alpha-\beta) & -\sin (\alpha+\beta) \cos (\alpha-\beta) & \sin (\alpha-\beta) \cos (\alpha+\beta) \\
\sin (\alpha+\beta) \sin (\alpha-\beta) & \cos (\alpha+\beta) \cos (\alpha-\beta) & \sin (\alpha-\beta) \cos (\alpha+\beta) & -\sin (\alpha+\beta) \cos (\alpha-\beta) \\
\sin (\alpha+\beta) \cos (\alpha-\beta) & -\sin (\alpha-\beta) \cos (\alpha+\beta) & \cos (\alpha+\beta) \cos (\alpha-\beta) & \sin (\alpha+\beta) \sin (\alpha-\beta) \\
-\sin (\alpha-\beta) \cos (\alpha+\beta) & \sin (\alpha+\beta) \cos (\alpha-\beta) & \sin (\alpha+\beta) \sin (\alpha-\beta) & \cos (\alpha+\beta) \cos (\alpha-\beta)
\end{array}\right).
```

```{code-cell}
result = R12(pi - t - phi, pi / 2 - t) * B_ab(0, t) * R12(phi - pi, -pi / 2)
result = simplify(result)
result
```

On the other hand, it can be proven that

```{math}
B^{\prime}\left(\alpha, \beta\right) = Man\left(\alpha, \alpha+\frac{\pi}{2}, \beta, \beta+\frac{\pi}{2}\right).
```

Thus, the desired operation can be expressed through the decomposition in Eq. [](#eq:std-bs-derivation-decomposition).

+++

## Feedforward

+++

### Feedforward within teleportation circuit

Define {math}`\boldsymbol{f}_\text{tel} \left(\theta_1, \theta_2; m_1, m_2\right)` in Eq. [](#eq:teleportation-ff).

```{code-cell}
def f_tel(_t1, _t2, _m1, _m2):
    return -(
        Matrix([
            [cos(_t2), cos(_t1)],
            [sin(_t2), sin(_t1)],
        ])
        * Matrix([_m1, _m2])
        * sqrt(2)
        / sin(_t2 - _t1)
    )


f_tel(t1, t2, m1, m2)
```

(sec:macronode-ff-derivation)=

### Feedforward within macronode circuit

Derive Eq. [](#eq:macronode-ff).

```{code-cell}
_XXPP = Bu * Matrix([
    f_tel(tbk, tak, mbk, mak)[0],
    f_tel(tdk, tck, mdk, mck)[0],
    f_tel(tbk, tak, mbk, mak)[1],
    f_tel(tdk, tck, mdk, mck)[1],
])


def f_mac_1(_ta, _tb, _tc, _td, _ma, _mb, _mc, _md):
    return Matrix([_XXPP[0], _XXPP[2]]).subs([
        (tak, _ta),
        (tbk, _tb),
        (tck, _tc),
        (tdk, _td),
        (mak, _ma),
        (mbk, _mb),
        (mck, _mc),
        (mdk, _md),
    ])


def f_mac_N(_ta, _tb, _tc, _td, _ma, _mb, _mc, _md):
    return Matrix([_XXPP[1], _XXPP[3]]).subs([
        (tak, _ta),
        (tbk, _tb),
        (tck, _tc),
        (tdk, _td),
        (mak, _ma),
        (mbk, _mb),
        (mck, _mc),
        (mdk, _md),
    ])


f_mac_1(tak, tbk, tck, tdk, mak, mbk, mck, mdk), f_mac_N(tak, tbk, tck, tdk, mak, mbk, mck, mdk)
```

```{code-cell}
X_k_1 = f_mac_1(tak, tbk, tck, tdk, mak, mbk, mck, mdk)[0]
P_k_1 = f_mac_1(tak, tbk, tck, tdk, mak, mbk, mck, mdk)[1]
X_k_N = f_mac_N(tak, tbk, tck, tdk, mak, mbk, mck, mdk)[0]
P_k_N = f_mac_N(tak, tbk, tck, tdk, mak, mbk, mck, mdk)[1]

Matrix([[X_k_1], [P_k_1]]), Matrix([[X_k_N], [P_k_N]])
```

(sec:numerical-ff-derivation)=

### Numerical feedforward

Derive {math}`\boldsymbol{f}^{+1}_\text{num} \left(\boldsymbol{\theta}^k, \boldsymbol{m}^k, \boldsymbol{\theta}^{k+1}\right)` and {math}`\boldsymbol{f}^{+N}_\text{num} \left(\boldsymbol{\theta}^k, \boldsymbol{m}^k, \boldsymbol{\theta}^{k+N}\right)` in {ref}`sec:numerical-feedforward`.

```{code-cell}
def f_num_1(_tak, _tbk, _tck, _tdk, _mak, _mbk, _mck, _mdk, _tak1, _tbk1, _tck1, _tdk1):
    return trigsimp(
        Matrix([
            [sin(tak1), 0, 0, 0, cos(tak1), 0, 0, 0],
            [0, sin(tbk1), 0, 0, 0, cos(tbk1), 0, 0],
            [0, 0, sin(tck1), 0, 0, 0, cos(tck1), 0],
            [0, 0, 0, sin(tdk1), 0, 0, 0, cos(tdk1)],
        ])
        * Ad
        * Matrix([0, X_k_1, 0, 0, 0, P_k_1, 0, 0])
    ).subs([
        (tak, _tak),
        (tbk, _tbk),
        (tck, _tck),
        (tdk, _tdk),
        (mak, _mak),
        (mbk, _mbk),
        (mck, _mck),
        (mdk, _mdk),
        (tak1, _tak1),
        (tbk1, _tbk1),
        (tck1, _tck1),
        (tdk1, _tdk1),
    ])


def f_num_N(_tak, _tbk, _tck, _tdk, _mak, _mbk, _mck, _mdk, _takN, _tbkN, _tckN, _tdkN):
    return trigsimp(
        Matrix([
            [sin(takN), 0, 0, 0, cos(takN), 0, 0, 0],
            [0, sin(tbkN), 0, 0, 0, cos(tbkN), 0, 0],
            [0, 0, sin(tckN), 0, 0, 0, cos(tckN), 0],
            [0, 0, 0, sin(tdkN), 0, 0, 0, cos(tdkN)],
        ])
        * Ad
        * Matrix([0, 0, 0, X_k_N, 0, 0, 0, P_k_N])
    ).subs([
        (tak, _tak),
        (tbk, _tbk),
        (tck, _tck),
        (tdk, _tdk),
        (mak, _mak),
        (mbk, _mbk),
        (mck, _mck),
        (mdk, _mdk),
        (takN, _takN),
        (tbkN, _tbkN),
        (tckN, _tckN),
        (tdkN, _tdkN),
    ])


(
    f_num_1(tak, tbk, tck, tdk, mak, mbk, mck, mdk, tak1, tbk1, tck1, tdk1),
    f_num_N(tak, tbk, tck, tdk, mak, mbk, mck, mdk, takN, tbkN, tckN, tdkN),
)
```

(sec:initialization-ff-derivation)=

### Feedforward immediately after initialization

Define readout measured values in Eq. [](#eq:readout-measured-values).

```{code-cell}
def M(_ma, _mb, _mc, _md):
    return (
        Matrix([
            [1, -1, 1, -1],
            [1, 1, 1, 1],
            [-1, 1, 1, -1],
            [-1, -1, 1, 1],
        ])
        * Matrix([_ma, _mb, _mc, _md])
        / 2
    )


M(mak, mbk, mck, mdk)
```

Define {math}`\mathscr{X}_k^{+1} \left(\theta_a^k, \boldsymbol{m}^k \right)`, {math}`\mathscr{P}_k^{+1} \left(\theta_a^k, \boldsymbol{m}^k \right)`, {math}`\mathscr{X}_k^{+N} \left(\theta_c^k, \boldsymbol{m}^k \right)`, and {math}`\mathscr{P}_k^{+N} \left(\theta_c^k, \boldsymbol{m}^k \right)` in Eq. [](#eq:initialization-macronode-ff).

```{code-cell}
iX_k_1 = -M(mak, mbk, mck, mdk)[0] * sin(tak)
iP_k_1 = M(mak, mbk, mck, mdk)[0] * cos(tak)
iX_k_N = -M(mak, mbk, mck, mdk)[2] * sin(tck)
iP_k_N = M(mak, mbk, mck, mdk)[2] * cos(tck)

Matrix([[iX_k_1], [iP_k_1]]), Matrix([[iX_k_N], [iP_k_N]])
```

Derive Eq. [](#eq:initialization-numerical-ff).

```{code-cell}
def i_num_1(_tak, _mak, _mbk, _mck, _mdk, _tak1, _tbk1, _tck1, _tdk1):
    return trigsimp(
        Matrix([
            [sin(tak1), 0, 0, 0, cos(tak1), 0, 0, 0],
            [0, sin(tbk1), 0, 0, 0, cos(tbk1), 0, 0],
            [0, 0, sin(tck1), 0, 0, 0, cos(tck1), 0],
            [0, 0, 0, sin(tdk1), 0, 0, 0, cos(tdk1)],
        ])
        * Ad
        * Matrix([0, iX_k_1, 0, 0, 0, iP_k_1, 0, 0])
    ).subs([
        (tak, _tak),
        (mak, _mak),
        (mbk, _mbk),
        (mck, _mck),
        (mdk, _mdk),
        (tak1, _tak1),
        (tbk1, _tbk1),
        (tck1, _tck1),
        (tdk1, _tdk1),
    ])


def i_num_N(_tck, _mak, _mbk, _mck, _mdk, _takN, _tbkN, _tckN, _tdkN):
    return trigsimp(
        Matrix([
            [sin(takN), 0, 0, 0, cos(takN), 0, 0, 0],
            [0, sin(tbkN), 0, 0, 0, cos(tbkN), 0, 0],
            [0, 0, sin(tckN), 0, 0, 0, cos(tckN), 0],
            [0, 0, 0, sin(tdkN), 0, 0, 0, cos(tdkN)],
        ])
        * Ad
        * Matrix([0, 0, 0, iX_k_N, 0, 0, 0, iP_k_N])
    ).subs([
        (tck, _tck),
        (mak, _mak),
        (mbk, _mbk),
        (mck, _mck),
        (mdk, _mdk),
        (takN, _takN),
        (tbkN, _tbkN),
        (tckN, _tckN),
        (tdkN, _tdkN),
    ])


i_num_1(tak, mak, mbk, mck, mdk, tak1, tbk1, tck1, tdk1), i_num_N(tck, mak, mbk, mck, mdk, takN, tbkN, tckN, tdkN)
```

(sec:displacement-derivation)=

## Displacement

Derive Eq. [](#eq:numerical-displacement).

```{code-cell}
dX_k_1 = symbols(r"\mathbb{X}_k^{+1}")
dP_k_1 = symbols(r"\mathbb{P}_k^{+1}")  # noqa: RUF027
dX_k_N = symbols(r"\mathbb{X}_k^{+N}")
dP_k_N = symbols(r"\mathbb{P}_k^{+N}")  # noqa: RUF027

Matrix([[dX_k_1], [dP_k_1]]), Matrix([[dX_k_N], [dP_k_N]])
```

```{code-cell}
def d_num_1(_dX_k_1, _dP_k_1, _tak1, _tbk1, _tck1, _tdk1):
    return trigsimp(
        Matrix([
            [sin(tak1), 0, 0, 0, cos(tak1), 0, 0, 0],
            [0, sin(tbk1), 0, 0, 0, cos(tbk1), 0, 0],
            [0, 0, sin(tck1), 0, 0, 0, cos(tck1), 0],
            [0, 0, 0, sin(tdk1), 0, 0, 0, cos(tdk1)],
        ])
        * Ad
        * Matrix([0, dX_k_1, 0, 0, 0, dP_k_1, 0, 0])
    ).subs([
        (dX_k_1, _dX_k_1),
        (dP_k_1, _dP_k_1),
        (tak1, _tak1),
        (tbk1, _tbk1),
        (tck1, _tck1),
        (tdk1, _tdk1),
    ])


def d_num_N(_dX_k_N, _dP_k_N, _takN, _tbkN, _tckN, _tdkN):
    return trigsimp(
        Matrix([
            [sin(takN), 0, 0, 0, cos(takN), 0, 0, 0],
            [0, sin(tbkN), 0, 0, 0, cos(tbkN), 0, 0],
            [0, 0, sin(tckN), 0, 0, 0, cos(tckN), 0],
            [0, 0, 0, sin(tdkN), 0, 0, 0, cos(tdkN)],
        ])
        * Ad
        * Matrix([0, 0, 0, dX_k_N, 0, 0, 0, dP_k_N])
    ).subs([
        (dX_k_N, _dX_k_N),
        (dP_k_N, _dP_k_N),
        (takN, _takN),
        (tbkN, _tbkN),
        (tckN, _tckN),
        (tdkN, _tdkN),
    ])


d_num_1(dX_k_1, dP_k_1, tak1, tbk1, tck1, tdk1), d_num_N(dX_k_N, dP_k_N, takN, tbkN, tckN, tdkN)
```
