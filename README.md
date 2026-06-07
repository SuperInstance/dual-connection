# dual-connection

**Dual affine connections on statistical manifolds — the geometry behind information.**

[![crates.io](https://img.shields.io/crates/v/dual-connection.svg)](https://crates.io/crates/dual-connection)
[![docs.rs](https://docs.rs/dual-connection/badge.svg)](https://docs.rs/dual-connection)
[![license](https://img.shields.io/crates/l/dual-connection.svg)](https://github.com/SuperInstance/dual-connection)

Every statistical model is a curved surface in the space of probability distributions. The Fisher information metric gives this surface its geometry — distances, angles, volumes. But to do calculus on a curved surface, you need more than a metric. You need a **connection**: a rule for how to differentiate vector fields, how to parallel-transport vectors along curves, and how to define "straight lines" (geodesics).

Information geometry reveals something beautiful: on exponential families, there isn't one natural connection — there are **two**. The **e-connection** (exponential) and the **m-connection** (mixture) are dual to each other, and they create a depth perception for statistical manifolds the way your two eyes create depth perception for the visual world. Alone, each connection is flat — its geodesics are straight lines in the right coordinate system. Together, they encode the curvature of the Fisher metric through their duality.

This is not abstract mathematics for its own sake. The e/m duality is what makes variational inference tractable, what connects maximum entropy to maximum likelihood, and what gives the KL divergence its fundamental role in machine learning. The Pythagorean theorem — yes, it generalizes — tells you when you can decompose statistical distances additively. Flatness tells you when a statistical model is "simple" in a precise information-theoretic sense.

This crate implements the full geometric toolkit: connections, their duals, curvature, torsion, parallel transport, and the divergences that emerge from dual flatness. It is designed to be correct, well-documented, and a genuine reference for anyone learning information geometry.

---

## The Metaphor: Two Eyes on a Curved World

Imagine standing on the surface of a sphere. With your left eye, you see straight lines that curve northward — these are the e-geodesics. With your right eye, you see straight lines that curve southward — these are the m-geodesics. Neither eye alone can tell you the surface is curved; each sees its own perfectly flat world. But when you open both eyes, you perceive the curvature through the discrepancy between the two views.

This is exactly what happens on a statistical manifold:
- The **e-connection** is flat in the natural (canonical) parameterization of an exponential family
- The **m-connection** is flat in the expectation parameterization
- The **Fisher metric** is the overlap between them — the curvature emerges from their disagreement

The α-connection family interpolates between them: α = +1 is the e-connection, α = −1 is the m-connection, and α = 0 is the Levi-Civita connection (the one that actually "sees" the curvature).

---

## Quick Start

```rust
use dual_connection::connection::Point;
use dual_connection::dual::{normal_e_connection, normal_m_connection, normal_alpha_connection};
use dual_connection::curvature::CurvatureTensor;
use dual_connection::divergence::kl_divergence_normal;

// A point on the normal distribution manifold: N(μ=0, σ=1)
let p = Point::new(vec![0.0, 1.0]);

// The e-connection and m-connection
let e_conn = normal_e_connection();
let m_conn = normal_m_connection();

// Both are flat — zero curvature
let e_curv = CurvatureTensor::from_connection(&e_conn, &p, 1e-5);
let m_curv = CurvatureTensor::from_connection(&m_conn, &p, 1e-5);
assert!(e_curv.is_flat(0.1));
assert!(m_curv.is_flat(0.1));

// The Levi-Civita connection (α=0) sees the curvature
let lc_conn = normal_alpha_connection(0.0);
let lc_curv = CurvatureTensor::from_connection(&lc_conn, &p, 1e-5);
assert!(!lc_curv.is_flat(0.01));

// KL divergence between two normal distributions
let q = Point::new(vec![1.0, 2.0]); // N(μ=1, σ=2)
let kl = kl_divergence_normal(&p, &q);
println!("KL(N(0,1) ‖ N(1,4)) = {kl:.6}");
```

---

## Architecture

```
                        ┌──────────────────┐
                        │   dual-connection │
                        └────────┬─────────┘
                                 │
         ┌───────────┬───────────┼───────────┬───────────┐
         │           │           │           │           │
    ┌────┴────┐ ┌────┴────┐ ┌───┴────┐ ┌────┴────┐ ┌────┴────┐
    │connection│ │  dual   │ │parallel│ │curvature│ │torsion  │
    │         │ │         │ │transport│ │         │ │         │
    │ Point   │ │ Metric  │ │        │ │ Riemann │ │ Torsion │
    │ Tangent │ │ α-conn  │ │ e vs m │ │ Ricci   │ │ Tensor  │
    │ Curve   │ │ e-conn  │ │compare │ │ Scalar  │ │ zero?   │
    │ Affine  │ │ m-conn  │ │holonomy│ │ flat?   │ │         │
    │ Connect │ │         │ │        │ │         │ │         │
    └────┬────┘ └────┬────┘ └───┬────┘ └────┬────┘ └────┬────┘
         │           │           │           │           │
         └───────────┴───────────┼───────────┴───────────┘
                                 │
                          ┌──────┴──────┐
                          │ divergence  │
                          │             │
                          │ KL / Canon  │
                          │ Bregman     │
                          │ α-divergence│
                          │ Pythagorean │
                          └─────────────┘
```

---

## Modules

| Module | Purpose | Key Types |
|--------|---------|-----------|
| `connection` | Affine connections, points, curves, parallel transport | `Point`, `TangentVector`, `Curve`, `AffineConnection` |
| `dual` | Dual connections, α-connection family, Fisher metric | `Metric`, `normal_alpha_connection`, `normal_e_connection`, `normal_m_connection` |
| `parallel_transport` | Transport vectors along curves, compare e vs m, holonomy | `transport_between`, `compare_e_m_transport`, `holonomy_angle` |
| `curvature` | Riemann curvature tensor, Ricci tensor, scalar curvature, flatness | `CurvatureTensor`, `RicciTensor`, `normal_scalar_curvature` |
| `torsion` | Torsion tensor, torsion-free verification, custom torsion | `TorsionTensor`, `is_torsion_free`, `with_torsion` |
| `divergence` | KL divergence, Bregman divergence, canonical divergence, Pythagorean theorem | `BregmanDivergence`, `PythagoreanResult`, `alpha_divergence_normal` |

---

## Mathematical Foundations

### Affine Connections

An affine connection ∇ on an n-dimensional manifold is specified by its **Christoffel symbols** Γᵏᵢⱼ in local coordinates. The connection defines:

- **Covariant derivative**: ∇ₓY = (∂ᵢYᵏ + Γᵏᵢⱼ Yʲ)Xⁱ ∂/∂xᵏ
- **Parallel transport**: dvᵏ/dt + Γᵏᵢⱼ(dxⁱ/dt)vʲ = 0
- **Geodesics**: d²xᵏ/dt² + Γᵏᵢⱼ(dxⁱ/dt)(dxʲ/dt) = 0

### The α-Connection Family

On a statistical manifold with Fisher metric gᵢⱼ, the α-connection is:

```
(Γᵏᵢⱼ)α = (1+α)/2 · Γᵏᵢⱼ(e) + (1−α)/2 · Γᵏᵢⱼ(m)
```

Special cases:
- **α = +1**: e-connection (exponential). Flat in natural parameters.
- **α = −1**: m-connection (mixture). Flat in expectation parameters.
- **α = 0**: Levi-Civita connection. The unique torsion-free metric connection.

### Duality

The dual connection ∇* of ∇ with respect to the metric g satisfies:

```
X · g(Y, Z) = g(∇ₓY, Z) + g(Y, ∇*ₓZ)
```

The dual of the α-connection is the (−α)-connection. This means:
- The dual of the e-connection is the m-connection
- The dual of the Levi-Civita connection is itself (self-dual)

### Curvature

The Riemann curvature tensor:

```
Rᵏˡᵢⱼ = ∂ᵢΓᵏˡⱼ − ∂ⱼΓᵏˡᵢ + ΓᵏᵢₘΓᵐˡⱼ − ΓᵏⱼₘΓᵐˡᵢ
```

A connection is **flat** iff R = 0 everywhere. For exponential families:
- The e-connection and m-connection are both flat
- The Levi-Civita connection has constant negative curvature on the normal manifold
- **Flatness = exponential family**: A statistical manifold is an exponential family iff it admits a flat connection

### Torsion

```
Tᵏᵢⱼ = Γᵏᵢⱼ − Γᵏⱼᵢ
```

All α-connections are torsion-free (symmetric in lower indices). Non-zero torsion indicates a connection that encodes more than just the metric structure.

### Canonical Divergence and the Pythagorean Theorem

When a manifold admits dual flat connections (∇ flat, ∇* flat), there exists a **canonical divergence** D(p‖q) satisfying:

```
D(p‖q) = ψ(θₚ) + φ(η_q) − θₚ · η_q
```

where ψ is the log-partition function and φ is the dual potential.

**Generalized Pythagorean Theorem**: If the e-geodesic from p to q is orthogonal (in the Fisher metric) to the m-geodesic from q to r, then:

```
D(p‖r) = D(p‖q) + D(q‖r)
```

For the normal distribution, the canonical divergence equals the KL divergence.

---

## The Normal Distribution Manifold

The primary worked example in this crate is the 2-dimensional manifold of normal distributions N(μ, σ²), parameterized by (μ, σ).

**Fisher metric** in (μ, σ) coordinates:
```
g = | 1/σ²    0   |
    |  0     2/σ² |
```

**Natural parameters**: (η₁, η₂) = (μ/σ², −1/(2σ²))
**Expectation parameters**: (μ₁, μ₂) = (μ, μ² + σ²)

**Christoffel symbols** of the α-connection:
```
Γ¹₁₂ = Γ¹₂₁ = −(1+α)/σ
Γ²₁₁ = (1−α)/(2σ)
Γ²₂₂ = −(1+2α)/σ
```
All others are zero.

**Key results**:
- e-connection (α=+1): Γ¹₁₂ = −2/σ, Γ²₁₁ = 0, Γ²₂₂ = −3/σ → **flat**
- m-connection (α=−1): Γ¹₁₂ = 0, Γ²₁₁ = 1/σ, Γ²₂₂ = 1/σ → **flat**
- Levi-Civita (α=0): Γ¹₁₂ = −1/σ, Γ²₁₁ = 1/(2σ), Γ²₂₂ = −1/σ → **curved** (R = −1)

---

## API Examples

### Computing Christoffel Symbols

```rust
use dual_connection::connection::Point;
use dual_connection::dual::normal_alpha_connection;

let conn = normal_alpha_connection(0.3);
let p = Point::new(vec![1.0, 2.0]); // N(μ=1, σ=2)

let gamma = conn.christoffel_at(&p);
// gamma[k][i][j] = Γᵏᵢⱼ at (μ=1, σ=2)
println!("Γ¹₁₂ = {:.6}", gamma[0][0][1]); // -(1+0.3)/2 = -0.65
println!("Γ²₁₁ = {:.6}", gamma[1][0][0]); // (1-0.3)/(2*2) = 0.175
println!("Γ²₂₂ = {:.6}", gamma[1][1][1]); // -(1+0.6)/2 = -0.8
```

### Checking Flatness via Curvature

```rust
use dual_connection::connection::Point;
use dual_connection::dual::normal_alpha_connection;
use dual_connection::curvature::CurvatureTensor;

let p = Point::new(vec![0.0, 1.0]);

for alpha in [-1.0, 0.0, 1.0] {
    let conn = normal_alpha_connection(alpha);
    let curv = CurvatureTensor::from_connection(&conn, &p, 1e-5);
    println!("α={:+.1}: flat={}, max|R|={:.6}",
        alpha, curv.is_flat(0.1), curv.max_abs());
}
// α=-1.0: flat=true,  max|R|≈0
// α=+0.0: flat=false, max|R|>0
// α=+1.0: flat=true,  max|R|≈0
```

### KL Divergence and the Pythagorean Theorem

```rust
use dual_connection::connection::Point;
use dual_connection::divergence::{kl_divergence_normal, pythagorean_check};

let p = Point::new(vec![0.0, 1.0]); // N(0, 1)
let q = Point::new(vec![0.0, 2.0]); // N(0, 4)
let r = Point::new(vec![0.0, 3.0]); // N(0, 9)

// KL divergence
let kl = kl_divergence_normal(&p, &q);
println!("KL(N(0,1)‖N(0,4)) = {:.6}", kl);

// Pythagorean check
let result = pythagorean_check(&p, &q, &r);
println!("D(p‖r) = {:.6}", result.d_pr);
println!("D(p‖q) + D(q‖r) = {:.6}", result.rhs);
println!("Pythagorean? {}", result.is_pythagorean);
```

### Parallel Transport: e vs m

```rust
use dual_connection::connection::{Point, TangentVector};
use dual_connection::parallel_transport::compare_e_m_transport;

let start = Point::new(vec![0.0, 1.0]);
let end = Point::new(vec![1.0, 2.0]);
let v = TangentVector::at(start.clone(), vec![1.0, 0.0]);

let (e_result, m_result) = compare_e_m_transport(&start, &end, &v, 500);
println!("e-transport: [{:.6}, {:.6}]", e_result.components[0], e_result.components[1]);
println!("m-transport: [{:.6}, {:.6}]", m_result.components[0], m_result.components[1]);
// Different! The discrepancy encodes the curvature of the Levi-Civita connection.
```

### Parameter Conversions

```rust
use dual_connection::connection::Point;
use dual_connection::divergence::{to_natural, from_natural, to_expectation, from_expectation};

let p = Point::new(vec![1.0, 2.0]); // N(μ=1, σ=2)

// Natural parameters: (μ/σ², −1/(2σ²)) = (0.25, −0.125)
let eta = to_natural(&p);
println!("η = [{:.4}, {:.4}]", eta[0], eta[1]);

// Expectation parameters: (μ, μ²+σ²) = (1, 5)
let mu = to_expectation(&p);
println!("μ = [{:.4}, {:.4}]", mu[0], mu[1]);

// Round-trip
let p2 = from_natural(&eta);
assert!((p2.coords[0] - 1.0).abs() < 1e-12);
```

---

## Design Decisions

### Why dynamic dispatch (`Rc<dyn Fn>`) for connections?

Connections are fundamentally functions Γ(p, k, i, j) → ℝ. The alternative — a trait object or enum — would make it harder to define connections as closures that capture the α parameter. `Rc<dyn Fn>` is the most natural representation.

### Why `Vec<f64>` instead of const generics?

Serde does not derive `Serialize`/`Deserialize` for arrays `[T; N]` with generic `N`. Using `Vec<f64>` avoids this limitation while keeping the API clean. The normal distribution manifold is 2-dimensional, so the overhead is negligible.

### Why no external dependencies beyond serde?

This crate implements the mathematical formulas directly. There is no need for linear algebra libraries (the matrices are 2×2), no need for special functions, and no need for auto-differentiation (we use finite differences for the general case and analytical formulas for the normal manifold).

### Why analytical Christoffel symbols for the normal manifold?

The α-connection Christoffel symbols for N(μ, σ) in (μ, σ) coordinates have simple closed-form expressions. Using these instead of numerical differentiation gives machine-precision accuracy and makes the tests deterministic.

The Christoffel symbols are derived by transforming from the flat coordinate systems:
- e-connection: flat in natural parameters (η₁, η₂) = (μ/σ², −1/(2σ²))
- m-connection: flat in expectation parameters (μ₁, μ₂) = (μ, μ² + σ²)

Using the transformation rule Γᵏᵢⱼ = (∂xᵏ/∂yᵃ)(∂²yᵃ/∂xⁱ∂xʲ) where y is the flat coordinate system.

---

## Running Tests

```bash
# Run all 62 tests
cargo test

# Run with output
cargo test -- --nocapture

# Format check
cargo fmt --check

# Lint
cargo clippy -- -D warnings
```

---

## Crate Features

- **Edition 2024** — uses the latest Rust edition
- **Zero runtime dependencies** — only `serde` (with `derive`)
- **62 comprehensive tests** — covering all modules with analytical verification
- **Serializable types** — all public data types derive `Serialize` + `Deserialize`

---

## References

- Amari, S. & Nagaoka, H. (2000). *Methods of Information Geometry*. AMS/Oxford University Press.
- Amari, S. (2016). *Information Geometry and Its Applications*. Springer.
- Murray, M.K. & Rice, J.W. (1993). *Differential Geometry and Statistics*. Chapman & Hall.

---

## License

Licensed under either of [Apache License, Version 2.0](LICENSE-APACHE) or [MIT license](LICENSE-MIT) at your option.

Unless you explicitly state otherwise, any contribution intentionally submitted for inclusion in this crate by you, as defined in the Apache-2.0 license, shall be dual licensed as above, without any additional terms or conditions.
