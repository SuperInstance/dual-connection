#![allow(clippy::needless_range_loop)]
//! Riemann curvature tensor, Ricci tensor, and scalar curvature.
//!
//! The curvature tensor measures how parallel transport around an infinitesimal
//! loop fails to return a vector to itself. For a flat connection (e.g., the
//! e-connection on an exponential family), all curvature components vanish.

use crate::connection::{AffineConnection, Point};
use crate::dual::{Metric, normal_alpha_connection, normal_metric};

/// Curvature tensor components at a point.
#[derive(Debug, Clone)]
pub struct CurvatureTensor {
    /// Rᵏˡᵢⱼ indexed as `[k][l][i][j]`.
    pub components: Vec<Vec<Vec<Vec<f64>>>>,
    /// The point where this tensor is evaluated.
    pub point: Point,
    /// Dimensionality.
    pub dim: usize,
}

impl CurvatureTensor {
    /// Compute the Riemann curvature tensor numerically from a connection.
    ///
    /// Uses:
    /// ```text
    /// Rᵏˡᵢⱼ = ∂ᵢΓᵏˡⱼ − ∂ⱼΓᵏˡᵢ + ΓᵏᵢₘΓᵐˡⱼ − ΓᵏⱼₘΓᵐˡᵢ
    /// ```
    pub fn from_connection(conn: &AffineConnection, point: &Point, h: f64) -> Self {
        let n = point.dim();
        let christoffel = &conn.christoffel;
        let mut r = vec![vec![vec![vec![0.0; n]; n]; n]; n];

        for k in 0..n {
            for l in 0..n {
                for i in 0..n {
                    for j in 0..n {
                        // ∂ᵢΓᵏˡⱼ via central difference
                        let mut pp = point.clone();
                        let mut pm = point.clone();
                        pp.coords[i] += h;
                        pm.coords[i] -= h;
                        let d_gamma_klj_di =
                            (christoffel(&pp, k, l, j) - christoffel(&pm, k, l, j)) / (2.0 * h);

                        // ∂ⱼΓᵏˡᵢ
                        let mut pp = point.clone();
                        let mut pm = point.clone();
                        pp.coords[j] += h;
                        pm.coords[j] -= h;
                        let d_gamma_kli_dj =
                            (christoffel(&pp, k, l, i) - christoffel(&pm, k, l, i)) / (2.0 * h);

                        // Γᵏᵢₘ Γᵐˡⱼ − Γᵏⱼₘ Γᵐˡᵢ
                        let mut q1 = 0.0;
                        let mut q2 = 0.0;
                        for m in 0..n {
                            q1 += christoffel(point, k, i, m) * christoffel(point, m, l, j);
                            q2 += christoffel(point, k, j, m) * christoffel(point, m, l, i);
                        }

                        r[k][l][i][j] = d_gamma_klj_di - d_gamma_kli_dj + q1 - q2;
                    }
                }
            }
        }

        Self {
            components: r,
            point: point.clone(),
            dim: n,
        }
    }

    /// Get Rᵏˡᵢⱼ.
    pub fn get(&self, k: usize, l: usize, i: usize, j: usize) -> f64 {
        self.components[k][l][i][j]
    }

    /// Is the curvature zero (flat) within tolerance?
    pub fn is_flat(&self, tol: f64) -> bool {
        self.components
            .iter()
            .flatten()
            .flatten()
            .flatten()
            .all(|&v| v.abs() < tol)
    }

    /// Maximum absolute curvature component.
    pub fn max_abs(&self) -> f64 {
        self.components
            .iter()
            .flatten()
            .flatten()
            .flatten()
            .map(|v| v.abs())
            .fold(0.0_f64, f64::max)
    }

    /// Compute the Ricci tensor Rᵢⱼ = Rᵏᵢₖⱼ.
    pub fn ricci(&self) -> RicciTensor {
        let n = self.dim;
        let mut r = vec![vec![0.0; n]; n];
        for i in 0..n {
            for j in 0..n {
                for k in 0..n {
                    r[i][j] += self.components[k][i][k][j];
                }
            }
        }
        RicciTensor {
            components: r,
            dim: n,
        }
    }
}

/// Ricci tensor.
#[derive(Debug, Clone)]
pub struct RicciTensor {
    /// Rᵢⱼ components.
    pub components: Vec<Vec<f64>>,
    /// Dimensionality.
    pub dim: usize,
}

impl RicciTensor {
    /// Compute scalar curvature R = gⁱʲ Rᵢⱼ.
    pub fn scalar_curvature(&self, g_inv: &[Vec<f64>]) -> f64 {
        let mut s = 0.0;
        for i in 0..self.dim {
            for j in 0..self.dim {
                s += g_inv[i][j] * self.components[i][j];
            }
        }
        s
    }
}

/// Convenience: curvature tensor for α-connection on the normal manifold.
pub fn normal_alpha_curvature(alpha: f64, point: &Point) -> CurvatureTensor {
    let conn = normal_alpha_connection(alpha);
    CurvatureTensor::from_connection(&conn, point, 1e-5)
}

/// Scalar curvature for α-connection on the normal manifold at (μ, σ).
pub fn normal_scalar_curvature(alpha: f64, point: &Point) -> f64 {
    let conn = normal_alpha_connection(alpha);
    let curv = CurvatureTensor::from_connection(&conn, point, 1e-5);
    let ricci = curv.ricci();
    let g = normal_metric(point);
    let m = Metric::new("fisher", g);
    let g_inv = m.inverse();
    ricci.scalar_curvature(&g_inv)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn e_connection_is_flat() {
        let conn = normal_alpha_connection(1.0);
        let p = Point::new(vec![0.0, 2.0]);
        let curv = CurvatureTensor::from_connection(&conn, &p, 1e-5);
        assert!(curv.is_flat(0.1), "e-connection should be flat");
    }

    #[test]
    fn m_connection_is_flat() {
        let conn = normal_alpha_connection(-1.0);
        let p = Point::new(vec![0.0, 2.0]);
        let curv = CurvatureTensor::from_connection(&conn, &p, 1e-5);
        assert!(curv.is_flat(0.1), "m-connection should be flat");
    }

    #[test]
    fn levi_civita_not_flat() {
        let conn = normal_alpha_connection(0.0);
        let p = Point::new(vec![0.0, 2.0]);
        let curv = CurvatureTensor::from_connection(&conn, &p, 1e-5);
        assert!(!curv.is_flat(0.01), "Levi-Civita should not be flat");
    }

    #[test]
    fn levi_civita_scalar_curvature_negative() {
        let p = Point::new(vec![0.0, 2.0]);
        let r = normal_scalar_curvature(0.0, &p);
        assert!(r < -0.1, "Scalar curvature should be negative, got {r}");
    }

    #[test]
    fn e_scalar_curvature_near_zero() {
        let p = Point::new(vec![0.0, 2.0]);
        let r = normal_scalar_curvature(1.0, &p);
        assert!(
            r.abs() < 0.5,
            "e-connection scalar curvature should be ~0, got {r}"
        );
    }

    #[test]
    fn m_scalar_curvature_near_zero() {
        let p = Point::new(vec![0.0, 2.0]);
        let r = normal_scalar_curvature(-1.0, &p);
        assert!(
            r.abs() < 0.5,
            "m-connection scalar curvature should be ~0, got {r}"
        );
    }

    #[test]
    fn flat_connection_zero_curvature() {
        let conn = AffineConnection::new("flat", |_, _, _, _| 0.0);
        let p = Point::new(vec![0.0, 0.0]);
        let curv = CurvatureTensor::from_connection(&conn, &p, 1e-5);
        assert!(curv.is_flat(1e-10));
    }

    #[test]
    fn curvature_tensor_shape() {
        let conn = normal_alpha_connection(0.5);
        let p = Point::new(vec![1.0, 2.0]);
        let curv = CurvatureTensor::from_connection(&conn, &p, 1e-5);
        assert_eq!(curv.components.len(), 2);
        assert_eq!(curv.components[0][0][0].len(), 2);
    }

    #[test]
    fn ricci_tensor_shape() {
        let conn = normal_alpha_connection(0.0);
        let p = Point::new(vec![0.0, 2.0]);
        let curv = CurvatureTensor::from_connection(&conn, &p, 1e-5);
        let ricci = curv.ricci();
        assert_eq!(ricci.components.len(), 2);
        assert_eq!(ricci.components[0].len(), 2);
    }

    #[test]
    fn max_abs_flat_is_zero() {
        let conn = AffineConnection::new("flat", |_, _, _, _| 0.0);
        let p = Point::origin(2);
        let curv = CurvatureTensor::from_connection(&conn, &p, 1e-5);
        assert!(curv.max_abs() < 1e-10);
    }
}
