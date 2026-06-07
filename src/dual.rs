#![allow(clippy::needless_range_loop)]
//! Dual connection and the α-connection family.
//!
//! Given an affine connection ∇ and a Riemannian metric *g*, the **dual
//! connection** ∇* satisfies:
//!
//! ```text
//! X · g(Y, Z) = g(∇ₓY, Z) + g(Y, ∇*ₓZ)
//! ```
//!
//! The **α-connection** family interpolates between the e-connection (α = +1)
//! and the m-connection (α = −1) via a single parameter α ∈ [−1, 1].

use serde::{Deserialize, Serialize};

use crate::connection::{AffineConnection, Point};

/// A Riemannian metric tensor on an *n*-dimensional manifold.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Metric {
    /// Components of the metric tensor gᵢⱼ stored row-major.
    pub inner: Vec<Vec<f64>>,
    /// Human-readable name.
    pub name: String,
}

impl Metric {
    /// Create a named metric from its components.
    pub fn new(name: impl Into<String>, inner: Vec<Vec<f64>>) -> Self {
        Self {
            inner,
            name: name.into(),
        }
    }

    /// Dimensionality.
    pub fn dim(&self) -> usize {
        self.inner.len()
    }

    /// Evaluate g(X, Y) = gᵢⱼ Xⁱ Yʲ.
    pub fn inner_product(&self, x: &[f64], y: &[f64]) -> f64 {
        let n = self.dim();
        let mut sum = 0.0;
        for i in 0..n {
            for j in 0..n {
                sum += self.inner[i][j] * x[i] * y[j];
            }
        }
        sum
    }

    /// Numerically invert the metric to get gⁱʲ.
    pub fn inverse(&self) -> Vec<Vec<f64>> {
        let n = self.dim();
        let mut aug = vec![vec![0.0; 2 * n]; n];
        for i in 0..n {
            for j in 0..n {
                aug[i][j] = self.inner[i][j];
            }
            aug[i][n + i] = 1.0;
        }
        for col in 0..n {
            let mut max_row = col;
            let mut max_val = aug[col][col].abs();
            for row in (col + 1)..n {
                if aug[row][col].abs() > max_val {
                    max_val = aug[row][col].abs();
                    max_row = row;
                }
            }
            aug.swap(col, max_row);
            let pivot = aug[col][col];
            for j in 0..(2 * n) {
                aug[col][j] /= pivot;
            }
            for row in 0..n {
                if row == col {
                    continue;
                }
                let factor = aug[row][col];
                for j in 0..(2 * n) {
                    aug[row][j] -= factor * aug[col][j];
                }
            }
        }
        (0..n)
            .map(|i| (0..n).map(|j| aug[i][n + j]).collect())
            .collect()
    }
}

/// Fisher information metric for the normal distribution N(μ, σ²) in (μ, σ) coordinates.
///
/// Returns the 2×2 metric tensor:
/// ```text
/// g = | 1/σ²    0    |
///     |  0     2/σ²  |
/// ```
pub fn normal_metric(point: &Point) -> Vec<Vec<f64>> {
    let sigma = point.coords[1];
    let s2 = sigma * sigma;
    vec![vec![1.0 / s2, 0.0], vec![0.0, 2.0 / s2]]
}

/// Build the α-connection for the normal distribution manifold in (μ, σ) coordinates.
///
/// Analytically computed Christoffel symbols:
/// ```text
/// Γ¹₁₁ = 0
/// Γ¹₁₂ = Γ¹₂₁ = (α − 1) / σ
/// Γ¹₂₂ = 0
/// Γ²₁₁ = 1 / (2σ)
/// Γ²₁₂ = Γ²₂₁ = 0
/// Γ²₂₂ = (2α − 1) / σ
/// ```
pub fn normal_alpha_connection(alpha: f64) -> AffineConnection {
    let name = format!("α-connection (α={alpha})");
    AffineConnection::new(name, move |point: &Point, k: usize, i: usize, j: usize| {
        let sigma = point.coords[1];
        // Correct α-connection Christoffel symbols for N(μ,σ) in (μ,σ) coords.
        // Derived by transforming from flat coordinate systems:
        //   e-connection is flat in natural params (η₁, η₂) = (μ/σ², −1/(2σ²))
        //   m-connection is flat in expectation params (μ₁, μ₂) = (μ, μ²+σ²)
        // α-connection = ((1+α)/2)·Γ_e + ((1−α)/2)·Γ_m
        match (k, i, j) {
            (0, 0, 1) | (0, 1, 0) => -(1.0 + alpha) / sigma,
            (1, 0, 0) => (1.0 - alpha) / (2.0 * sigma),
            (1, 1, 1) => -(1.0 + 2.0 * alpha) / sigma,
            _ => 0.0,
        }
    })
}

/// The e-connection (α = +1) on the normal distribution manifold.
pub fn normal_e_connection() -> AffineConnection {
    normal_alpha_connection(1.0)
}

/// The m-connection (α = −1) on the normal distribution manifold.
pub fn normal_m_connection() -> AffineConnection {
    normal_alpha_connection(-1.0)
}

/// The Levi-Civita connection (α = 0) on the normal distribution manifold.
pub fn normal_levi_civita() -> AffineConnection {
    normal_alpha_connection(0.0)
}

/// Compute the dual α-connection: the dual of α-connection is the (−α)-connection.
pub fn dual_alpha(alpha: f64) -> f64 {
    -alpha
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normal_metric_diagonal() {
        let p = Point::new(vec![0.0, 2.0]); // μ=0, σ=2
        let g = normal_metric(&p);
        assert!((g[0][0] - 0.25).abs() < 1e-12);
        assert!((g[1][1] - 0.5).abs() < 1e-12);
        assert!((g[0][1]).abs() < 1e-12);
    }

    #[test]
    fn metric_inverse() {
        let p = Point::new(vec![0.0, 2.0]);
        let g = normal_metric(&p);
        let m = Metric::new("fisher", g);
        let inv = m.inverse();
        assert!((inv[0][0] - 4.0).abs() < 1e-10);
        assert!((inv[1][1] - 2.0).abs() < 1e-10);
        assert!((inv[0][1]).abs() < 1e-10);
    }

    #[test]
    fn inner_product_standard() {
        let m = Metric::new("fisher", vec![vec![1.0, 0.0], vec![0.0, 1.0]]);
        let v = vec![1.0, 0.0];
        let ip = m.inner_product(&v, &v);
        assert!((ip - 1.0).abs() < 1e-12);
    }

    #[test]
    fn e_connection_christoffels() {
        let conn = normal_e_connection();
        let p = Point::new(vec![0.0, 1.0]); // σ=1
        let gamma = conn.christoffel_at(&p);
        // α=1: Γ⁰₀₁ = -(1+1)/1 = −2, Γ¹₀₀ = 0, Γ¹₁₁ = -(1+2)/1 = −3
        assert!((gamma[0][0][1] - (-2.0)).abs() < 1e-12);
        assert!((gamma[1][0][0] - 0.0).abs() < 1e-12);
        assert!((gamma[1][1][1] - (-3.0)).abs() < 1e-12);
    }

    #[test]
    fn m_connection_christoffels() {
        let conn = normal_m_connection();
        let p = Point::new(vec![0.0, 1.0]); // σ=1
        let gamma = conn.christoffel_at(&p);
        // α=−1: Γ⁰₀₁ = 0, Γ¹₀₀ = (1−(−1))/(2·1) = 1, Γ¹₁₁ = −(1−2)/1 = 1
        assert!((gamma[0][0][1] - 0.0).abs() < 1e-12);
        assert!((gamma[1][0][0] - 1.0).abs() < 1e-12);
        assert!((gamma[1][1][1] - 1.0).abs() < 1e-12);
    }

    #[test]
    fn levi_civita_christoffels() {
        let conn = normal_levi_civita();
        let p = Point::new(vec![0.0, 2.0]); // σ=2
        let gamma = conn.christoffel_at(&p);
        // α=0: Γ⁰₀₁ = −1/2, Γ¹₀₀ = 1/4, Γ¹₁₁ = −1/2
        assert!((gamma[0][0][1] - (-0.5)).abs() < 1e-12);
        assert!((gamma[1][0][0] - 0.25).abs() < 1e-12);
        assert!((gamma[1][1][1] - (-0.5)).abs() < 1e-12);
    }

    #[test]
    fn all_alpha_connections_are_torsion_free() {
        for alpha in [-1.0, -0.5, 0.0, 0.5, 1.0] {
            let conn = normal_alpha_connection(alpha);
            let p = Point::new(vec![1.0, 2.0]);
            assert!(
                conn.is_torsion_free(&p, 1e-12),
                "α={alpha} not torsion-free"
            );
        }
    }

    #[test]
    fn alpha_connection_symmetry() {
        for alpha in [-1.0, 0.0, 1.0] {
            let conn = normal_alpha_connection(alpha);
            let p = Point::new(vec![1.0, 3.0]);
            let gamma = conn.christoffel_at(&p);
            let n = gamma.len();
            for k in 0..n {
                for i in 0..n {
                    for j in 0..n {
                        assert!(
                            (gamma[k][i][j] - gamma[k][j][i]).abs() < 1e-12,
                            "Not symmetric: Γ^{}{}_{} vs Γ^{}{}_{}",
                            k,
                            i,
                            j,
                            k,
                            j,
                            i
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn dual_alpha_flips_sign() {
        assert_eq!(dual_alpha(1.0), -1.0);
        assert_eq!(dual_alpha(-1.0), 1.0);
        assert_eq!(dual_alpha(0.0), 0.0);
    }

    #[test]
    fn alpha_connection_at_different_sigma() {
        let conn = normal_alpha_connection(0.5);
        let p1 = Point::new(vec![0.0, 1.0]);
        let p2 = Point::new(vec![0.0, 2.0]);
        let g1 = conn.christoffel_at(&p1);
        let g2 = conn.christoffel_at(&p2);
        // Γ⁰₀₁(α=0.5) = −(1+0.5)/σ = −1.5/σ
        assert!((g1[0][0][1] - (-1.5)).abs() < 1e-12);
        assert!((g2[0][0][1] - (-0.75)).abs() < 1e-12);
    }

    #[test]
    fn metric_identity_check() {
        let p = Point::new(vec![0.0, 3.0]);
        let g = normal_metric(&p);
        let m = Metric::new("fisher", g);
        let inv = m.inverse();
        for i in 0..2 {
            let mut sum = 0.0;
            for j in 0..2 {
                sum += m.inner[i][j] * inv[j][i];
            }
            assert!((sum - 1.0).abs() < 1e-10, "g·g⁻¹ ≠ I at row {i}");
        }
    }
}
