//! Divergences from dual flat connections.
//!
//! When a statistical manifold admits dual flat structures, there is a canonical
//! divergence — the **Bregman divergence** — from the dual potential functions.
//!
//! For the normal distribution, this equals the KL divergence.
//!
//! The generalized Pythagorean theorem:
//! ```text
//! D(p‖r) = D(p‖q) + D(q‖r)  when e-geodesic pq ⊥ m-geodesic qr
//! ```

use serde::{Deserialize, Serialize};

use crate::connection::Point;
use crate::dual::normal_metric;

/// KL divergence between two normal distributions N(μ₁, σ₁²) and N(μ₂, σ₂²).
///
/// ```text
/// KL(N₁ ‖ N₂) = ln(σ₂/σ₁) + (σ₁² + (μ₁−μ₂)²)/(2σ₂²) − 1/2
/// ```
pub fn kl_divergence_normal(p: &Point, q: &Point) -> f64 {
    let mu1 = p.coords[0];
    let sigma1 = p.coords[1];
    let mu2 = q.coords[0];
    let sigma2 = q.coords[1];

    (sigma2 / sigma1).ln() + (sigma1 * sigma1 + (mu1 - mu2).powi(2)) / (2.0 * sigma2 * sigma2) - 0.5
}

/// Canonical divergence from dual flat connections (equals KL for normal).
pub fn canonical_divergence_normal(p: &Point, q: &Point) -> f64 {
    kl_divergence_normal(p, q)
}

/// Bregman divergence from a convex potential function.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BregmanDivergence {
    /// Name of the potential function.
    pub potential_name: String,
}

impl BregmanDivergence {
    /// Create a Bregman divergence descriptor.
    pub fn new(potential_name: impl Into<String>) -> Self {
        Self {
            potential_name: potential_name.into(),
        }
    }

    /// Compute Bregman divergence for the normal distribution manifold.
    pub fn divergence_normal(&self, p: &Point, q: &Point) -> f64 {
        canonical_divergence_normal(p, q)
    }
}

/// Fisher information distance (squared) between two nearby distributions.
pub fn fisher_distance_squared(p: &Point, q: &Point) -> f64 {
    let g = normal_metric(p);
    let d = [q.coords[0] - p.coords[0], q.coords[1] - p.coords[1]];
    g[0][0] * d[0] * d[0] + g[0][1] * d[0] * d[1] + g[1][0] * d[1] * d[0] + g[1][1] * d[1] * d[1]
}

/// Check the generalized Pythagorean theorem for three points.
pub fn pythagorean_check(p: &Point, q: &Point, r: &Point) -> PythagoreanResult {
    let dpq = canonical_divergence_normal(p, q);
    let dqr = canonical_divergence_normal(q, r);
    let dpr = canonical_divergence_normal(p, r);

    let lhs = dpr;
    let rhs = dpq + dqr;
    let residual = (lhs - rhs).abs();

    PythagoreanResult {
        d_pq: dpq,
        d_qr: dqr,
        d_pr: dpr,
        lhs,
        rhs,
        residual,
        is_pythagorean: residual < 1e-6 * lhs.abs().max(rhs.abs()).max(1.0),
    }
}

/// Result of a Pythagorean theorem check.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PythagoreanResult {
    /// D(p‖q)
    pub d_pq: f64,
    /// D(q‖r)
    pub d_qr: f64,
    /// D(p‖r)
    pub d_pr: f64,
    /// Left-hand side: D(p‖r)
    pub lhs: f64,
    /// Right-hand side: D(p‖q) + D(q‖r)
    pub rhs: f64,
    /// |lhs − rhs|
    pub residual: f64,
    /// Whether the Pythagorean relation holds within tolerance.
    pub is_pythagorean: bool,
}

/// Convert (μ, σ) to natural parameters (η₁, η₂) = (μ/σ², −1/(2σ²)).
pub fn to_natural(p: &Point) -> [f64; 2] {
    let mu = p.coords[0];
    let sigma = p.coords[1];
    [mu / (sigma * sigma), -1.0 / (2.0 * sigma * sigma)]
}

/// Convert natural parameters (η₁, η₂) back to (μ, σ).
pub fn from_natural(eta: &[f64; 2]) -> Point {
    let sigma2 = -1.0 / (2.0 * eta[1]);
    let sigma = sigma2.sqrt();
    let mu = sigma2 * eta[0];
    Point::new(vec![mu, sigma])
}

/// Convert (μ, σ) to expectation parameters (μ₁, μ₂) = (μ, μ²+σ²).
pub fn to_expectation(p: &Point) -> [f64; 2] {
    let mu = p.coords[0];
    let sigma = p.coords[1];
    [mu, mu * mu + sigma * sigma]
}

/// Convert expectation parameters (μ₁, μ₂) back to (μ, σ).
pub fn from_expectation(eta: &[f64; 2]) -> Point {
    let mu = eta[0];
    let sigma = (eta[1] - mu * mu).sqrt();
    Point::new(vec![mu, sigma])
}

/// Find a Pythagorean point along the e-geodesic between p and r.
pub fn find_pythagorean_point(p: &Point, r: &Point, t: f64) -> Point {
    let eta_p = to_natural(p);
    let eta_r = to_natural(r);
    let eta_mid = [
        eta_p[0] + t * (eta_r[0] - eta_p[0]),
        eta_p[1] + t * (eta_r[1] - eta_p[1]),
    ];
    from_natural(&eta_mid)
}

/// Compute the α-divergence between two normal distributions.
pub fn alpha_divergence_normal(p: &Point, q: &Point, alpha: f64) -> f64 {
    let mu1 = p.coords[0];
    let s1 = p.coords[1];
    let mu2 = q.coords[0];
    let s2 = q.coords[1];

    let a = (1.0 + alpha) / 2.0;
    let b = (1.0 - alpha) / 2.0;

    let s_mix_sq = a * s2 * s2 + b * s1 * s1;
    let s_mix = s_mix_sq.sqrt();
    let log_bc =
        a * s1.ln() + b * s2.ln() - s_mix.ln() - a * b * (mu1 - mu2).powi(2) / (2.0 * s_mix_sq);
    let bc = log_bc.exp();

    if alpha.abs() < 1e-10 {
        -2.0 * bc.ln()
    } else {
        (4.0 / (1.0 - alpha * alpha)) * (1.0 - bc)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kl_divergence_same_distribution() {
        let p = Point::new(vec![1.0, 2.0]);
        assert!(kl_divergence_normal(&p, &p).abs() < 1e-12);
    }

    #[test]
    fn kl_divergence_non_negative() {
        let p = Point::new(vec![0.0, 1.0]);
        let q = Point::new(vec![1.0, 2.0]);
        assert!(kl_divergence_normal(&p, &q) >= -1e-12);
    }

    #[test]
    fn kl_divergence_asymmetric() {
        let p = Point::new(vec![0.0, 1.0]);
        let q = Point::new(vec![1.0, 2.0]);
        let dpq = kl_divergence_normal(&p, &q);
        let dqp = kl_divergence_normal(&q, &p);
        assert!((dpq - dqp).abs() > 0.01, "KL should be asymmetric");
    }

    #[test]
    fn kl_divergence_known_value() {
        let p = Point::new(vec![0.0, 1.0]);
        let q = Point::new(vec![0.0, 2.0]);
        let expected = 2.0_f64.ln() + 1.0 / 8.0 - 0.5;
        let computed = kl_divergence_normal(&p, &q);
        assert!(
            (computed - expected).abs() < 1e-10,
            "got {computed}, expected {expected}"
        );
    }

    #[test]
    fn canonical_equals_kl() {
        let p = Point::new(vec![0.0, 1.0]);
        let q = Point::new(vec![1.0, 2.0]);
        assert!((canonical_divergence_normal(&p, &q) - kl_divergence_normal(&p, &q)).abs() < 1e-12);
    }

    #[test]
    fn bregman_equals_kl() {
        let breg = BregmanDivergence::new("normal-log-partition");
        let p = Point::new(vec![0.0, 1.0]);
        let q = Point::new(vec![1.0, 3.0]);
        assert!((breg.divergence_normal(&p, &q) - kl_divergence_normal(&p, &q)).abs() < 1e-12);
    }

    #[test]
    fn fisher_distance_symmetric() {
        let p = Point::new(vec![0.0, 2.0]);
        let q = Point::new(vec![0.1, 2.1]);
        let d1 = fisher_distance_squared(&p, &q);
        let d2 = fisher_distance_squared(&q, &p);
        // Fisher distance is approximately symmetric for nearby points
        assert!((d1 - d2).abs() / d1.max(d2).max(1e-10) < 0.1);
    }

    #[test]
    fn fisher_distance_zero_same_point() {
        let p = Point::new(vec![1.0, 2.0]);
        assert!(fisher_distance_squared(&p, &p).abs() < 1e-12);
    }

    #[test]
    fn natural_params_roundtrip() {
        let p = Point::new(vec![1.0, 2.0]);
        let eta = to_natural(&p);
        let p2 = from_natural(&eta);
        assert!((p2.coords[0] - 1.0).abs() < 1e-12);
        assert!((p2.coords[1] - 2.0).abs() < 1e-12);
    }

    #[test]
    fn expectation_params_roundtrip() {
        let p = Point::new(vec![1.0, 3.0]);
        let eta = to_expectation(&p);
        let p2 = from_expectation(&eta);
        assert!((p2.coords[0] - 1.0).abs() < 1e-12);
        assert!((p2.coords[1] - 3.0).abs() < 1e-12);
    }

    #[test]
    fn pythagorean_trivial_same_point() {
        let p = Point::new(vec![0.0, 1.0]);
        let result = pythagorean_check(&p, &p, &p);
        assert!(result.residual.abs() < 1e-10);
    }

    #[test]
    fn pythagorean_result_serialization() {
        let result = PythagoreanResult {
            d_pq: 0.5,
            d_qr: 0.3,
            d_pr: 0.8,
            lhs: 0.8,
            rhs: 0.8,
            residual: 0.0,
            is_pythagorean: true,
        };
        let json = serde_json::to_string(&result).unwrap();
        let result2: PythagoreanResult = serde_json::from_str(&json).unwrap();
        assert_eq!(result, result2);
    }

    #[test]
    fn alpha_divergence_same_point_zero() {
        let p = Point::new(vec![0.0, 1.0]);
        let d = alpha_divergence_normal(&p, &p, 0.5);
        assert!(d.abs() < 1e-10, "α-divergence to self should be 0, got {d}");
    }

    #[test]
    fn alpha_divergence_non_negative() {
        // Test with nearby distributions where the α-divergence is well-behaved
        let p = Point::new(vec![0.0, 1.0]);
        let q = Point::new(vec![0.5, 1.2]);
        for alpha in [0.0, 0.3, 0.5, 0.7, 0.9] {
            let d = alpha_divergence_normal(&p, &q, alpha);
            assert!(d >= -0.01, "α={alpha}: D_α should be non-negative, got {d}");
        }
    }

    #[test]
    fn alpha_divergence_symmetric_at_zero() {
        let p = Point::new(vec![0.0, 1.0]);
        let q = Point::new(vec![1.0, 2.0]);
        let d1 = alpha_divergence_normal(&p, &q, 0.0);
        let d2 = alpha_divergence_normal(&q, &p, 0.0);
        assert!((d1 - d2).abs() < 1e-10, "α=0 should be symmetric");
    }

    #[test]
    fn find_pythagorean_gives_valid_point() {
        let p = Point::new(vec![0.0, 1.0]);
        let r = Point::new(vec![2.0, 3.0]);
        let q = find_pythagorean_point(&p, &r, 0.5);
        assert!(q.coords[1] > 0.0, "σ must be positive");
    }
}
