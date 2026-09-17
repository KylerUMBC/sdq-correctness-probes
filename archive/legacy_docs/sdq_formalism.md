# SDQ Formalism

## Trajectory Space

Let $\Gamma = \{\gamma \mid \gamma = (h_0, h_1, \dots, h_T), \; h_t \in \mathbb{R}^d\}$
be the space of observed hidden-state trajectories produced by an LLM.

## Decomposition

Each observed hidden state factors as:

$$h_t = F(z_t, u_t)$$

where:
- $z_t$ — latent semantic state (reasoning content)
- $u_t$ — surface realization / gauge (prompt wording, format, syntax)

## Local Surface Transports

For trajectories $\gamma_i, \gamma_j$, define local transport:

$$\Delta h_{\tau(t)}^{(j)} \approx G_t^{(i \to j)} \Delta h_t^{(i)}$$

where $\tau$ is a monotone time alignment and $G_t^{(i \to j)}$ is a local
transport operator conditioned on trajectory context.

## Equivalence Relation

$\gamma_i \sim \gamma_j$ if there exists:
- a monotone alignment $\tau$
- a family of local transports $G_t^{(i \to j)}$

such that they preserve the same latent semantic dynamics.

## Groupoid Structure

SDQ is framed as a **trajectory groupoid**, not a group action:
- **objects**: trajectories
- **morphisms**: local surface transports between compatible trajectories

This is necessary because transforms are local, state-dependent, and
conditional on prompt family / region of trajectory space.

## Generative Model

$$h_t = D_{u_t}(z_t) + \epsilon_t$$

where $D_{u_t}$ is a surface-dependent observation chart / decoder.

## Invariants

The SDQ quotient preserves:
1. **Motion**: direction classes, curvature, bottleneck passage
2. **Events**: premise loading, entity binding, rule application, answer commitment
3. **Dependencies**: semantic dependency structure across reasoning steps
