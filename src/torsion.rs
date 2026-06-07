#![allow(clippy::type_complexity)]
//! Torsion tensor of an affine connection.
//!
//! ```text
//! Tᵏᵢⱼ = Γᵏᵢⱼ − Γᵏⱼᵢ
//! ```
//!
//! A torsion-free connection (Levi-Civita, all α-connections) satisfies T ≡ 0.

use serde::{Deserialize, Serialize};

use crate::connection::{AffineConnection, Point};

/// Torsion tensor at a point.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TorsionTensor {
    /// Tᵏᵢⱼ components.
    pub components: Vec<Vec<Vec<f64>>>,
    /// Evaluation point.
    pub point: Point,
}

impl TorsionTensor {
    /// Compute the torsion tensor of a connection at a point.
    pub fn from_connection(conn: &AffineConnection, point: &Point) -> Self {
        let christoffel = &conn.christoffel;
        let n = point.dim();
        let components = (0..n)
            .map(|k| {
                (0..n)
                    .map(|i| {
                        (0..n)
                            .map(|j| christoffel(point, k, i, j) - christoffel(point, k, j, i))
                            .collect()
                    })
                    .collect()
            })
            .collect();

        Self {
            components,
            point: point.clone(),
        }
    }

    /// Get component Tᵏᵢⱼ.
    pub fn get(&self, k: usize, i: usize, j: usize) -> f64 {
        self.components[k][i][j]
    }

    /// Is the torsion zero (within tolerance)?
    pub fn is_zero(&self, tol: f64) -> bool {
        self.components
            .iter()
            .flatten()
            .flatten()
            .all(|&v| v.abs() < tol)
    }

    /// Maximum absolute torsion component.
    pub fn max_abs(&self) -> f64 {
        self.components
            .iter()
            .flatten()
            .flatten()
            .map(|v| v.abs())
            .fold(0.0_f64, f64::max)
    }

    /// Verify antisymmetry: Tᵏᵢⱼ = −Tᵏⱼᵢ.
    pub fn verify_antisymmetry(&self, tol: f64) -> bool {
        let n = self.components.len();
        for k in 0..n {
            for i in 0..n {
                for j in 0..n {
                    if (self.components[k][i][j] + self.components[k][j][i]).abs() > tol {
                        return false;
                    }
                }
            }
        }
        true
    }
}

/// Check whether a connection is torsion-free at a given point.
pub fn is_torsion_free(conn: &AffineConnection, point: &Point, tol: f64) -> bool {
    let t = TorsionTensor::from_connection(conn, point);
    t.is_zero(tol)
}

/// Build a connection with prescribed torsion by adding an antisymmetric part.
pub fn with_torsion(
    base: &AffineConnection,
    antisymmetric: Box<dyn Fn(&Point, usize, usize, usize) -> f64>,
    name: impl Into<String>,
) -> AffineConnection {
    let base_christoffel = base.christoffel.clone();
    AffineConnection::new(name, move |point, k, i, j| {
        base_christoffel(point, k, i, j) + antisymmetric(point, k, i, j)
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dual::{normal_alpha_connection, normal_e_connection, normal_m_connection};

    #[test]
    fn e_connection_torsion_free() {
        let conn = normal_e_connection();
        let p = Point::new(vec![1.0, 2.0]);
        let t = TorsionTensor::from_connection(&conn, &p);
        assert!(t.is_zero(1e-12));
    }

    #[test]
    fn m_connection_torsion_free() {
        let conn = normal_m_connection();
        let p = Point::new(vec![0.0, 3.0]);
        let t = TorsionTensor::from_connection(&conn, &p);
        assert!(t.is_zero(1e-12));
    }

    #[test]
    fn all_alpha_torsion_free() {
        for alpha in [-1.0, -0.5, 0.0, 0.5, 1.0] {
            let conn = normal_alpha_connection(alpha);
            let p = Point::new(vec![1.0, 2.0]);
            assert!(
                is_torsion_free(&conn, &p, 1e-12),
                "α={alpha} should be torsion-free"
            );
        }
    }

    #[test]
    fn torsion_tensor_antisymmetry() {
        let t = TorsionTensor {
            components: vec![
                vec![vec![0.0, 1.0], vec![-1.0, 0.0]],
                vec![vec![0.0, 2.0], vec![-2.0, 0.0]],
            ],
            point: Point::origin(2),
        };
        assert!(t.verify_antisymmetry(1e-12));
    }

    #[test]
    fn with_torsion_creates_nonzero_torsion() {
        let base = AffineConnection::new("base", |_, _, _, _| 0.0);
        let torsionous = with_torsion(
            &base,
            Box::new(|_, _k, i, j| {
                if i == 0 && j == 1 {
                    1.0
                } else if i == 1 && j == 0 {
                    -1.0
                } else {
                    0.0
                }
            }),
            "torsionous",
        );
        let p = Point::origin(2);
        let t = TorsionTensor::from_connection(&torsionous, &p);
        assert!(!t.is_zero(1e-12));
        assert!((t.get(0, 0, 1) - 2.0).abs() < 1e-12);
    }

    #[test]
    fn zero_connection_zero_torsion() {
        let conn = AffineConnection::new("zero", |_, _, _, _| 0.0);
        let p = Point::origin(2);
        let t = TorsionTensor::from_connection(&conn, &p);
        assert!(t.is_zero(1e-12));
    }

    #[test]
    fn torsion_max_abs_zero() {
        let conn = normal_alpha_connection(0.0);
        let p = Point::new(vec![0.0, 1.0]);
        let t = TorsionTensor::from_connection(&conn, &p);
        assert!(t.max_abs() < 1e-12);
    }

    #[test]
    fn torsion_serialization_roundtrip() {
        let conn = normal_alpha_connection(0.5);
        let p = Point::new(vec![1.0, 2.0]);
        let t = TorsionTensor::from_connection(&conn, &p);
        let json = serde_json::to_string(&t).unwrap();
        let t2: TorsionTensor = serde_json::from_str(&json).unwrap();
        assert_eq!(t, t2);
    }
}
