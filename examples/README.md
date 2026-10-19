# Thermal isolation bracket — problem formulation

The examples in this directory all solve one design problem: a composite bracket
that carries a mechanical load between a liquid-hydrogen tank and a fuselage
frame while leaking as little heat as possible. They differ only in dimension,
in whether the fibre orientation is a design field, and in whether a minimum
length scale is imposed.

Geometry and loads follow *Aerospace* **2026**, 13, 388 (Table 8 for the
bracket, Table 11 config C / 3 g lateral / outer row for the load, Table 1 for
the AS4/PEEK UD lamina). Units are mm / N / MPa / K / W throughout.

```
      top edge / face    T = T_hot = 293 K,  clamped   -- fuselage frame
                         passive solid band, 5% of LY
          design region  rho per element,
                         orientation per node
                         passive solid band, 5% of LY
   bottom edge / face    T = T_cold = 20 K,  traction  -- LH2 tank
      remaining edges    free and insulated
```

The two interfaces are both a thermal boundary and a mechanical one, so the
structure cannot simply disconnect them: heat and load share every path.

## The optimisation problem

$$
\begin{aligned}
\min_{\mathbf{x}} \quad & J(\mathbf{x}) \;=\; \frac{H(\mathbf{x})}{H_{\text{ref}}} \;+\; w_m V(\mathbf{x}) \\[4pt]
\text{s.t.} \quad & \frac{C(\mathbf{x})}{C_{\text{ref}}} \;\le\; \bar{C} \\[2pt]
& \frac{e(\mathbf{x})}{\bar{e}} \;\le\; 1 \\[2pt]
& \frac{g_s(\mathbf{x})}{\varepsilon} \;\le\; 1, \qquad \frac{g_v(\mathbf{x})}{\varepsilon} \;\le\; 1 \qquad \text{(geommetric file only)} \\[2pt]
& 0 \le \rho_e \le 1, \qquad -1 \le t_i \le 1
\end{aligned}
$$

with $w_m$ = `W_MASS`, $\bar{C}$ = `ENERGY_ALLOW`, $\bar{e}$ = `STRAIN_ALLOW`,
$\varepsilon$ = `LS_EPS` or `LS_MORPH_EPS`.

Every response is normalised by the **reference design**: the same domain filled
with solid material at the isotropic orientation, $\mathbf{a}_2 = \tfrac{1}{2}\mathbf{I}$
in 2-D (planar random) and $\mathbf{a}_2 = \tfrac{1}{3}\mathbf{I}$ in 3-D (random
in space). So $1.0$ always means "what a solid block of the same composite
achieves", and the limits are dimensionless multiples of it.

### Objective

The thermal term is the conduction energy of the solved temperature field,

$$
H \;=\; \tfrac{1}{2}\int_\Omega \nabla T \cdot \mathbf{k}(\rho, \mathbf{a}_2)\, \nabla T \; d\Omega ,
$$

and the mass term is the mean density over the **design** elements only,

$$
V \;=\; \frac{1}{|\Omega_d|}\sum_{e \in \Omega_d} \rho_e ,
$$

the passive bands being solid by construction and contributing only a constant.

For a two-terminal problem the transmitted heat follows directly,

$$
Q \;=\; \frac{2H}{\Delta T} \times t ,
\qquad \Delta T = T_{\text{hot}} - T_{\text{cold}},
$$

with the thickness factor $t$ present in 2-D, where the energy comes out per
unit thickness, and absent in 3-D, where it is already a volume integral.
Minimising $H$ therefore minimises the leak exactly. Runs report both.

$w_m = 1$ trades mass against heat one for one in units of their own references.

### Compliance constraint

$C$ is the weighted mean over load cases $i$ of the stored elastic energy, each
normalised by its own reference:

$$
C \;=\; \frac{1}{\sum_i w_i} \sum_i w_i \, \frac{U_i}{U_{\text{ref},i}},
\qquad
U_i \;=\; \tfrac{1}{2}\int_\Omega \boldsymbol{\varepsilon}_i : \mathbb{C}(\rho, \mathbf{a}_2) : \boldsymbol{\varepsilon}_i \; d\Omega .
$$

With no thermal expansion ($\alpha_\parallel = \alpha_\perp = 0$) this is exactly
half the compliance $\int_{\Gamma} \mathbf{t}\cdot\mathbf{u}$, so the constraint
is the standard normalised compliance constraint. Two load cases are defined —
tension and shear — and only those with $w_i > 0$ are built and solved. At
present only tension carries weight.

$\bar{C}$ is the single most consequential knob. The objective *wants* the
bracket cut in two, and this constraint is what forbids it. Measured behaviour:

- $\bar{C} = 1$ with fixed isotropic orientation admits exactly one design, the
  solid plate — nothing can beat the reference in stiffness, so nothing can be
  removed. The run correctly converges there.
- Slack limits (order $10^2$) leave the constraint inactive and let the optimiser
  spend the freedom on pathologies instead. Keep it active.

### Deformation constraint

$$
e \;=\; \frac{1}{C} \cdot \frac{\big\| \, \mathbb{C}_0 : \boldsymbol{\varepsilon} \, \big\|_p}{s_{\text{ref}}},
\qquad
\big\|f\big\|_p \;=\; \left( \frac{\sum_q \omega_q \, f_q^{\,p}}{\sum_q \omega_q} \right)^{1/p},
\qquad p = 12 .
$$

The field is the **unrelaxed** stress measure: $\mathbb{C}_0$ carries no density
weighting, so this is a strain measure wearing stress units. Divided by the
compliance it reports how *localised* the deformation is, not how much there is,
and that is what forbids a weak load path.

A stress constraint does **not** work here and was tried. The relaxed measure
carries $\rho^{\,q}$, which crushes the value exactly where the density is low,
so a bracket cut in two scores *lower* than an intact one. At $q = 0$ the density
leaves the field and what remains grows with how broken the structure is.

The **aggregation weight** carries the density, the field never does:

$$
\omega_q \;=\; \rho_{e(q)} \, (J w)_q .
$$

Unweighted, 52% of the p-norm came from elements below $\rho = 0.01$ — it was
measuring how freely the void deforms. Weighting the field by $\rho^{\,q}$
instead is the opposite failure. Weighting only the aggregation keeps the field
at $\mathbb{C}_0 : \boldsymbol{\varepsilon}$, so a grey bridge at $\rho = 0.31$
reports its full value while the void beside it is scaled out.

### Minimum length scale — only in `thermal_isolation_2d_aniso_geommetric.py`

The geometric constraints of Zhou, Lazarov, Wang & Sigmund, *CMAME* **293**
(2015) 266–282, plus a morphological alternative. `LS_METHOD` picks which pair
enters the constraint set; both are reported either way.

| method | what it polices | blind spot |
|---|---|---|
| `"geometric"` | stationary points of $\tilde\rho$, Eqs. (10–11): $\\{x : \bar\rho = 1,\; \nabla\tilde\rho = 0\\}$ | a member that *tapers* has no stationary point — its indicator is killed by $e^{-c\lvert\nabla\tilde\rho\rvert^2}$ and it may thin without limit. Measured, a strut tapering to 2 mm against an 8 mm target left $g_s$ inside its bound |
| `"morphological"` | opening / closing of the blueprint — asks the width question directly | says nothing about threshold robustness |

The morphological pair is the opening $\gamma$ and closing $\phi$ under a ball
of radius $w/2$, softened by log-sum-exp so both are differentiable:

$$
g_s = \big\| \max(\rho - \gamma(\rho),\, 0) \big\|_p ,
\qquad
g_v = \big\| \max(\phi(\rho) - \rho,\, 0) \big\|_p ,
$$

$$
\text{erode}(\rho) = -\tfrac{1}{k}\log\!\big( \mathbf{W} e^{-k\rho} \big),
\qquad
\text{dilate}(\rho) = \tfrac{1}{k}\log\!\big( \mathbf{W} e^{k\rho} \big).
$$

The aggregate is a p-norm and **not** a volume average. Averaged, a violation
confined to a few elements is divided by the whole domain: measured, the design
came out in eight pieces with a 1 mm neck against an 8 mm target while the
averaged constraint sat at 0.9998 of its bound.

The constraints switch on from $\beta \ge 12$ rather than at the start: on a grey
field the width of a "member" is not yet defined, and the paper is explicit that
switching on early finds a local minimum fast. Measured here, a design scoring
0.95–1.00 at $\beta = 2\text{–}4$ jumped to 1.77–2.14 the moment $\beta$ reached
8 — the low-$\beta$ evaluation carried no information.

**Measured: the filter alone very nearly delivers the target, and the constraint
overshoots it.** Without any length-scale constraint the narrowest member comes
out at 5.7–6.0 mm against a 6 mm target ($R = 6.63$ mm), and at 2.8 mm against a
2.71 mm target ($R = 3.0$ mm). With the constraint on it comes out at 7.2–8.0 mm
— 20–33% wider than asked — and costs 5.6–7.0% of the heat objective. Consider
whether the constraint earns its place before enabling it.

### Reported, not constrained

The von Mises p-norm ($p = 12$, relaxed with $q = 2$) and its pooled maximum are
logged in real MPa for information only.

## Design variables

$$
\mathbf{x} \;=\; \big[\, \underbrace{\rho_1 \dots \rho_{n_c}}_{\text{elements}} \;\big|\;
\underbrace{t_1, t_2, \dots}_{\text{nodes}} \,\big]
$$

The orientation half is present only when `ANISOTROPY` is on; otherwise the
orientation is pinned to the isotropic state and the design is density only.

| field | lives on | count | filter |
|---|---|---|---|
| $\rho$ | element centroids | $n_c$ | linear hat ("cone"), radius $R$ |
| orientation | nodes | 2 in 2-D ($t_1, t_2$), 4 in 3-D ($t_1, t_2, t_{12}, t_{13}$) | Helmholtz, radius $R_{\text{ori}}$ |

**Density is per element, not per node, and this matters.** With nodal density a
one-node-wide cut still interpolates to non-zero density inside the neighbouring
elements and retains 24% of the solid stiffness — the optimiser found exactly
that loophole and built structures that were visually severed yet satisfied the
compliance limit. Element density closes it.

Orientation is only filtered, never projected: projection has no meaning for a
field with no void state, and in 3-D the two sign variables must stay free to
pass through zero, which is where the off-diagonals vanish.

The three-field chain for density is the usual one,

$$
\rho \;\xrightarrow{\;\text{filter}\;}\; \tilde\rho
\;\xrightarrow{\;\text{project}\;}\; \bar\rho
\;\xrightarrow{\;\text{passive}\;}\; 1 \ \text{ on } \Omega_p ,
$$

$$
\bar\rho \;=\; \frac{\tanh(\beta\eta) + \tanh\!\big(\beta(\tilde\rho - \eta)\big)}
{\tanh(\beta\eta) + \tanh\!\big(\beta(1 - \eta)\big)},
\qquad \eta = 0.5 .
$$

`jnp.where` on the passive bands also kills their gradient, so those variables
are inert rather than merely overwritten.

## Material interpolation

Both physics use RAMP, with **opposite signs of $q$**:

$$
k(\rho) \;=\; k_{\min} + (1 - k_{\min})\,\frac{\rho}{1 + q_k (1 - \rho)},
\qquad q_k = -0.8,
$$

$$
E(\rho) \;=\; E_{\min} + (E_s - E_{\min})\,\frac{\rho}{1 + q_E (1 - \rho)},
\qquad q_E = 8.0 .
$$

At $\rho = 0.5$ the conductivity weight is $0.83$ while the stiffness weight is
$0.10$. Grey material therefore conducts nearly like solid while carrying almost
no load, which is what makes intermediate density unattractive to an optimiser
that wants stiffness and hates conduction.

### Orientation

Anisotropy enters through the second-order orientation tensor $\mathbf{a}_2$,
built from box variables so the bounds are simple and every admissible state is
reachable. Density and orientation are separable in both physics: the RAMP
weight multiplies the orientation-averaged property, it never enters the
orientation average itself.

$$
\mathbf{k}(\rho, \mathbf{a}_2) = w_k(\rho)\, \mathbf{k}(\mathbf{a}_2),
\qquad
\mathbb{C}(\rho, \mathbf{a}_2) = w_E(\rho)\, \mathbb{C}(\mathbf{a}_2),
$$

so the strain energy density the element actually integrates is

$$
\psi \;=\; \tfrac{1}{2}\, w_E(\rho)\;
\boldsymbol{\varepsilon} : \mathbb{C}(\mathbf{a}_2) : \boldsymbol{\varepsilon}.
$$

**Conductivity is linear in $\mathbf{a}_2$:**

$$
\mathbf{k}(\mathbf{a}_2) \;=\; k_\perp \mathbf{I} \;+\; (k_\parallel - k_\perp)\, \mathbf{a}_2 .
$$

The isotropic state therefore sits at the **arithmetic mean** of $k_\parallel$
and $k_\perp$, not between them geometrically:

| state | $k_{yy}$, W/(mm·K) | relative to isotropic |
|---|---|---|
| $\mathbf{a}_2 = \tfrac{1}{2}\mathbf{I}$ (isotropic, 2-D) | $2.53\times10^{-3}$ | $1.000$ |
| fibres along the heat path | $4.31\times10^{-3}$ | $1.701$ |
| fibres across it | $7.56\times10^{-4}$ | $0.299$ |

so the insulating headroom is a factor $3.35$ while the penalty for aligning with
the load is $1.70$.

**Stiffness is not linear in $\mathbf{a}_2$** — it needs the fourth-order
orientation tensor $\mathbf{a}_4$. The Advani–Tucker orientation average is

$$
\begin{aligned}
C_{ijkl}(\mathbf{a}_2, \mathbf{a}_4) \;=\;
& B_1\, a_{ijkl} \\
+\; & B_2\, \big( a_{ij}\delta_{kl} + a_{kl}\delta_{ij} \big) \\
+\; & B_3\, \big( a_{ik}\delta_{jl} + a_{il}\delta_{jk} + a_{jk}\delta_{il} + a_{jl}\delta_{ik} \big) \\
+\; & B_4\, \delta_{ij}\delta_{kl} \\
+\; & B_5\, \big( \delta_{ik}\delta_{jl} + \delta_{il}\delta_{jk} \big)
\end{aligned}
$$

with the invariants $B_n$ built from the lamina moduli in its principal axes.
In 2-D plane stress, with $\nu_{21} = \nu_{12} E_2 / E_1$ and
$m = 1/(1 - \nu_{12}\nu_{21})$,

$$
Q_{11} = m E_1, \quad Q_{22} = m E_2, \quad Q_{12} = m \nu_{21} E_1, \quad Q_{66} = G_{12},
$$

$$
B_1 = Q_{11} + Q_{22} - 2Q_{12} - 4Q_{66}, \quad
B_2 = Q_{12}, \quad
B_3 = Q_{66} - \tfrac{Q_{22}}{2}, \quad
B_4 = 0, \quad
B_5 = \tfrac{Q_{22}}{2},
$$

the $B_4$ term being absent because the plane-stress reduction folds the
bulk-like $\delta\delta$ contribution into $B_5$. In 3-D, with
$m = 1/\big[(1+\nu_{23})(1 - \nu_{23} - 2\nu_{12}\nu_{21})\big]$,

$$
\begin{aligned}
C_{1111} &= E_1 (1 - \nu_{23}^2)\, m, &
C_{1122} &= E_1 \nu_{21} (1 + \nu_{23})\, m, \\
C_{2222} &= E_2 (1 - \nu_{12}\nu_{21})\, m, &
C_{2233} &= E_2 (\nu_{23} + \nu_{12}\nu_{21})\, m, \\
C_{1212} &= G_{12}, &
\end{aligned}
$$

$$
\begin{aligned}
B_1 &= C_{1111} + C_{2222} - 2C_{1122} - 4C_{1212}, &
B_2 &= C_{1122} - C_{2233}, \\
B_3 &= C_{1212} + \tfrac{C_{2233} - C_{2222}}{2}, &
B_4 &= C_{2233}, \\
B_5 &= \tfrac{C_{2222} - C_{2233}}{2}. &
\end{aligned}
$$

Only five engineering constants are independent: the transverse shear modulus
$G_{23} = (C_{2222} - C_{2233})/2 = E_2 / \big(2(1+\nu_{23})\big)$ is fixed by
transverse isotropy.

$\mathbf{a}_4$ is not a design field, so it comes from the **hybrid
(Advani–Tucker) closure**, which blends the two limiting closures by a scalar
measure of distance from isotropy:

$$
\mathbf{a}_4 \;=\; (1 - f)\, \mathbf{a}_4^{\text{lin}} \;+\; f\, \mathbf{a}_4^{\text{quad}},
\qquad
f \;=\; 1 - d^{\,d} \det(\mathbf{a}_2),
$$

with $d^{\,d} = 4$ in 2-D and $27$ in 3-D, so $f = 0$ at isotropy — where the
linear closure is exact — and $f = 1$ at full alignment, where the quadratic one
is. The plain quadratic closure is the wrong default: at
$\mathbf{a}_2 = \tfrac{1}{2}\mathbf{I}$ it badly underestimates the in-plane
modulus. See item 2 of the 3-D file's docstring for the measured error in 3-D
and why the problem is still well posed there.

## Continuation and the optimiser

| stage | $\beta$ | iterations |
|---|---|---|
| 1 | 2 | 6 |
| 2 | 4 | 6 |
| 3 | 8 | 10 |
| 4 | 12 | 20 |
| 5 (final) | 16 | up to 50, stops on the test below |

Low $\beta$ gets few iterations on purpose: there the projection is nearly
linear, the design is a blurred blob, and the fine features it grows are not what
the sharp stages keep — iterations spent there only commit the topology early.

The final stage stops when the **spread of $J$ over a trailing window** of
$N = 10$ iterations, relative to its own mean, falls below $10^{-3}$ *and* the
compliance margin is within $10^{-3}$:

$$
\frac{\max_{i \in N} J_i - \min_{i \in N} J_i}{\big| \overline{J}_N \big|} < 10^{-3}
\quad \text{and} \quad
\frac{C}{\bar{C}} - 1 \le 10^{-3} .
$$

The window, rather than a consecutive-step test, because MMA does not descend
monotonically here — measured, $J$ bounced between 2.34 and 2.78 over ten
evaluations while still drifting down, so any "$|\mathrm{d}J|/|J|$ small $N$
times in a row" criterion either never fires or fires on a coincidence. A spread
cannot be satisfied by an oscillation and cannot be satisfied by a slow drift
either, since both widen it. Feasibility is checked alongside so a run that
stalls while still violating the limit is never reported as converged.

The optimiser is `feax.experimental.mma.GCMMA`. $\beta$ and the length-scale
switch are traced **arguments**, not closure constants, so all stages reuse one
compiled step. Each constraint carries its own elastic penalty $c_j$: MMA buys
violations at price $c_j$ and abandons any constraint whose multiplier exceeds
it, so the physical constraints are priced at $10^7$ and the length-scale pair, a
manufacturing preference, at $10^6$.

## The files

| file | dim | orientation | length scale | constraints |
|---|---|---|---|---|
| `thermal_isolation_2d.py` | 2-D plane stress | switchable | none | 2 |
| `thermal_isolation_2d_aniso_geommetric.py` | 2-D plane stress | switchable | geometric + morphological | 4 |
| `thermal_isolation_3d_aniso_uncoupled.py` | 3-D | switchable | none | 2 |

The 2-D pair is identical apart from the length-scale machinery, so a run of one
against the other isolates the effect of that constraint alone.

Current settings:

| | `2d` | `2d_aniso_geommetric` | `3d` |
|---|---|---|---|
| domain, mm | $150 \times 115 \times 4$ | $150 \times 115 \times 4$ | $50 \times 50 \times 50$ |
| element, mm | 1.0 | 1.0 | 1.5 |
| $\bar{C}$ = `ENERGY_ALLOW` | 1.5 | 2 | 8 |
| $w_m$ = `W_MASS` | 1 | 1 | 1 |
| `ANISOTROPY` | False | False | True |
| `MIN_WIDTH`, mm | 6 | 6 | 6 |
| $\bar{e}$ = `STRAIN_ALLOW` | 5.0 | 5.0 | 5.0 |

`SEED, PERTURB` seeds a random start: $\text{PERTURB} > 0$ adds element-wise
uniform noise of that amplitude to the initial $\rho = 0.1$. At
$\text{PERTURB} = 0.1$ the range is exactly $[0, 0.2]$, so the clip never fires
and the initial volume is preserved; the cone filter then leaves about 6%
modulation. A perturbed output directory gets an `_s<seed>p<amp>` suffix so it
cannot overwrite the control.

## Outputs

Each run writes to `outputs/<script>_C<limit>_m<W_MASS>[_iso][_s#p#]/`:

| file | contents |
|---|---|
| `density.npy` | final $\rho$, per element |
| `density_nodal.npy` | the same averaged onto nodes, for plotting |
| `orientation.npy` | final $\mathbf{a}_2$, per node |
| `history.csv` | every logged scalar, one row per evaluation |
| `design.png` / `history.png` | density field (2-D only) and convergence traces |
| `final.vtu` | density, temperature, displacement, stress, fibre director |
| `snapshots.h5` | design fields every `SNAPSHOT_EVERY` iterations, with every constant above as a root attribute |
| `snapshots_evolution.xdmf` | the same as a ParaView time series |

Fields in the `.vtu` are **masked by density**. The state equations are posed on
the whole domain — the void is a very weak material, not an absence — so a raw
temperature or stress field fills the void with values that are real solutions
and completely uninformative. Masked fields go out under the plain names and the
raw ones keep a `_raw` suffix. This is presentation only: the objective and the
constraints are already weighted by $k(\rho)$ and $E(\rho)$.

## Known limitations

- **Grey bridges.** The constraints as posed do not forbid a load path carried by
  intermediate density. A design has been observed that is two disconnected
  components at $\rho > 0.5$, joined across a five-element gap whose peak density
  is $0.33$, and that still satisfied a slack compliance limit. The deformation
  constraint and an active compliance limit both push against this, but neither
  rules it out.
- **The reported performance is flattered by a few percent.** About 5% of the
  heat comes from the $\rho \in [0.1, 0.5]$ transition band, which a manufactured
  part would not have.
- **Anisotropic runs rarely converge within the cap.** They routinely use all 92
  evaluations while isotropic runs converge in 53–81, so anisotropic results
  should be read as upper bounds on the achievable objective.
- The 3-D closure is inexact for planar-random states; see item 2 of that file's
  docstring for the measured error and why the problem is still well posed.
