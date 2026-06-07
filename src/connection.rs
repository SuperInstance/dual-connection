//! Affine connection on a differentiable manifold.
//!
//! An affine connection ∇ defines how to differentiate vector fields and
//! transport vectors along curves. On an *n*-dimensional manifold it is
//! specified by its Christoffel symbols Γᵏᵢⱼ in local coordinates.

#![allow(clippy::type_complexity)]
#![allow(clippy::needless_range_loop)]

use serde::{Deserialize, Serialize};
use std::rc::Rc;

/// A point on an *n*-dimensional manifold represented by its coordinates.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Point {
    /// Coordinate values.
    pub coords: Vec<f64>,
}

impl Point {
    /// Create a point from a vector of coordinates.
    pub fn new(coords: Vec<f64>) -> Self {
        Self { coords }
    }

    /// Create a point from an array.
    pub fn from_array(arr: &[f64]) -> Self {
        Self { coords: arr.to_vec() }
    }

    /// Zero origin of dimension `n`.
    pub fn origin(n: usize) -> Self {
        Self { coords: vec![0.0; n] }
    }

    /// Dimensionality.
    pub fn dim(&self) -> usize {
        self.coords.len()
    }
}

/// A tangent vector at a point on the manifold.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TangentVector {
    /// Base point.
    pub point: Point,
    /// Components in the coordinate basis.
    pub components: Vec<f64>,
}

impl TangentVector {
    /// Create a tangent vector at a given point.
    pub fn at(point: Point, components: Vec<f64>) -> Self {
        Self { point, components }
    }

    /// Tangent vector at the origin with given components.
    pub fn at_origin(components: Vec<f64>) -> Self {
        Self {
            point: Point::origin(components.len()),
            components,
        }
    }

    /// Dimensionality.
    pub fn dim(&self) -> usize {
        self.components.len()
    }

    /// Scale by a scalar.
    pub fn scale(&self, s: f64) -> Self {
        Self::at(self.point.clone(), self.components.iter().map(|c| c * s).collect())
    }

    /// Add two tangent vectors at the same point (panics if points differ or dims mismatch).
    pub fn add(&self, other: &TangentVector) -> Self {
        debug_assert_eq!(self.point, other.point, "tangent vectors must be at same point");
        let comps: Vec<f64> = self.components.iter().zip(&other.components).map(|(a, b)| a + b).collect();
        Self::at(self.point.clone(), comps)
    }

    /// Euclidean norm.
    pub fn norm(&self) -> f64 {
        self.components.iter().map(|c| c * c).sum::<f64>().sqrt()
    }
}

/// A smooth curve γ : [a, b] → M parametrized by sample points.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Curve {
    /// Sample points along the curve, in order.
    pub points: Vec<Point>,
    /// Parameter values corresponding to sample points (strictly increasing).
    pub parameters: Vec<f64>,
}

impl Curve {
    /// Create a curve from parameter-value / point pairs.
    pub fn new(pairs: Vec<(f64, Point)>) -> Self {
        let mut pairs = pairs;
        pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        let (parameters, points): (Vec<_>, Vec<_>) = pairs.into_iter().unzip();
        Self { points, parameters }
    }

    /// Linearly interpolate a point at parameter `t`.
    pub fn evaluate(&self, t: f64) -> Option<Point> {
        if self.parameters.len() < 2 || t < self.parameters[0] || t > *self.parameters.last().unwrap() {
            return None;
        }
        let idx = match self.parameters.binary_search_by(|v| v.partial_cmp(&t).unwrap()) {
            Ok(i) => return Some(self.points[i].clone()),
            Err(i) => i,
        };
        if idx == 0 || idx >= self.parameters.len() {
            return None;
        }
        let t0 = self.parameters[idx - 1];
        let t1 = self.parameters[idx];
        let frac = (t - t0) / (t1 - t0);
        let coords: Vec<f64> = (0..self.points[0].dim())
            .map(|k| self.points[idx - 1].coords[k] * (1.0 - frac) + self.points[idx].coords[k] * frac)
            .collect();
        Some(Point::new(coords))
    }

    /// Approximate tangent (velocity) at parameter `t` via central difference.
    pub fn tangent(&self, t: f64) -> Option<TangentVector> {
        let h = 1e-6;
        let p_plus = self.evaluate(t + h)?;
        let p_minus = self.evaluate(t - h)?;
        let dim = p_plus.dim();
        let comps: Vec<f64> = (0..dim)
            .map(|i| (p_plus.coords[i] - p_minus.coords[i]) / (2.0 * h))
            .collect();
        let base = self.evaluate(t)?;
        Some(TangentVector::at(base, comps))
    }
}

/// An affine connection on an *n*-dimensional manifold.
///
/// Encapsulated as a function that computes the Christoffel symbols
/// Γᵏᵢⱼ at a given point.
#[derive(Clone)]
pub struct AffineConnection {
    /// Compute Christoffel symbol Γᵏᵢⱼ at a point.
    pub christoffel: Rc<dyn Fn(&Point, usize, usize, usize) -> f64>,
    /// Human-readable name.
    pub name: String,
}

impl AffineConnection {
    /// Create a named affine connection from its Christoffel-symbol function.
    pub fn new(
        name: impl Into<String>,
        christoffel: impl Fn(&Point, usize, usize, usize) -> f64 + 'static,
    ) -> Self {
        Self {
            christoffel: Rc::new(christoffel),
            name: name.into(),
        }
    }

    /// Evaluate all Christoffel symbols at a point, returning `gamma[k][i][j] = Γᵏᵢⱼ`.
    pub fn christoffel_at(&self, point: &Point) -> Vec<Vec<Vec<f64>>> {
        let n = point.dim();
        (0..n)
            .map(|k| {
                (0..n)
                    .map(|i| (0..n).map(|j| (self.christoffel)(point, k, i, j)).collect())
                    .collect()
            })
            .collect()
    }

    /// Covariant derivative ∇ₓY of vector field Y in direction X at a point.
    ///
    /// In coordinates: (∇ₓY)ᵏ = ∂ᵢYᵏ Xⁱ + Γᵏᵢⱼ Xⁱ Yʲ
    pub fn covariant_derivative(
        &self,
        point: &Point,
        x: &[f64],
        y: &[f64],
        dy: &[Vec<f64>],
    ) -> Vec<f64> {
        let n = point.dim();
        (0..n)
            .map(|k| {
                let mut sum = 0.0_f64;
                for i in 0..n {
                    sum += dy[k][i] * x[i];
                    for j in 0..n {
                        sum += (self.christoffel)(point, k, i, j) * x[i] * y[j];
                    }
                }
                sum
            })
            .collect()
    }

    /// Is this connection torsion-free (Γᵏᵢⱼ = Γᵏⱼᵢ for all k, i, j)?
    pub fn is_torsion_free(&self, point: &Point, tol: f64) -> bool {
        let n = point.dim();
        for k in 0..n {
            for i in 0..n {
                for j in 0..n {
                    let diff = (self.christoffel)(point, k, i, j) - (self.christoffel)(point, k, j, i);
                    if diff.abs() > tol {
                        return false;
                    }
                }
            }
        }
        true
    }

    /// Numerical parallel transport of `v` along `curve` using `steps` steps.
    pub fn parallel_transport(
        &self,
        v: &TangentVector,
        curve: &Curve,
        steps: usize,
    ) -> TangentVector {
        let mut v = v.components.clone();
        let t_start = curve.parameters.first().copied().unwrap_or(0.0);
        let t_end = curve.parameters.last().copied().unwrap_or(1.0);
        let dt = (t_end - t_start) / steps as f64;
        let n = v.len();

        for step in 0..steps {
            let t = t_start + (step as f64 + 0.5) * dt;
            let Some(pt) = curve.evaluate(t) else { continue };
            let Some(tang) = curve.tangent(t) else { continue };
            let dx = &tang.components;

            let mut dv = vec![0.0; n];
            for k in 0..n {
                for i in 0..n {
                    for j in 0..n {
                        dv[k] -= (self.christoffel)(&pt, k, i, j) * dx[i] * v[j] * dt;
                    }
                }
            }
            for k in 0..n {
                v[k] += dv[k];
            }
        }

        let endpoint = curve.evaluate(t_end).unwrap_or_else(|| {
            curve.points.last().cloned().unwrap_or_else(|| Point::origin(n))
        });
        TangentVector::at(endpoint, v)
    }
}

/// Build a straight-line curve between two points using `steps` sample points.
pub fn straight_line_curve(p0: &Point, p1: &Point, steps: usize) -> Curve {
    let pairs: Vec<(f64, Point)> = (0..=steps)
        .map(|s| {
            let t = s as f64 / steps as f64;
            let coords: Vec<f64> = (0..p0.dim())
                .map(|i| p0.coords[i] * (1.0 - t) + p1.coords[i] * t)
                .collect();
            (t, Point::new(coords))
        })
        .collect();
    Curve::new(pairs)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn point_origin_is_zeroed() {
        let p = Point::origin(3);
        assert_eq!(p.coords, vec![0.0, 0.0, 0.0]);
    }

    #[test]
    fn tangent_vector_scale() {
        let tv = TangentVector::at_origin(vec![1.0, 2.0, 3.0]);
        let s = tv.scale(2.0);
        assert_eq!(s.components, vec![2.0, 4.0, 6.0]);
    }

    #[test]
    fn tangent_vector_add() {
        let p = Point::origin(2);
        let a = TangentVector::at(p.clone(), vec![1.0, 2.0]);
        let b = TangentVector::at(p, vec![3.0, 4.0]);
        let c = a.add(&b);
        assert_eq!(c.components, vec![4.0, 6.0]);
    }

    #[test]
    fn curve_evaluate_midpoint() {
        let c = Curve::new(vec![
            (0.0, Point::new(vec![0.0])),
            (1.0, Point::new(vec![10.0])),
        ]);
        let mid = c.evaluate(0.5).unwrap();
        assert!((mid.coords[0] - 5.0).abs() < 1e-10);
    }

    #[test]
    fn curve_evaluate_out_of_bounds() {
        let c = Curve::new(vec![
            (0.0, Point::new(vec![0.0])),
            (1.0, Point::new(vec![10.0])),
        ]);
        assert!(c.evaluate(-0.1).is_none());
        assert!(c.evaluate(1.1).is_none());
    }

    #[test]
    fn zero_connection_is_torsion_free() {
        let conn = AffineConnection::new("zero", |_, _, _, _| 0.0);
        let p = Point::new(vec![1.0, 2.0]);
        assert!(conn.is_torsion_free(&p, 1e-12));
    }

    #[test]
    fn covariant_derivative_flat() {
        let conn = AffineConnection::new("flat", |_, _, _, _| 0.0);
        let p = Point::new(vec![0.0, 0.0]);
        let x = vec![1.0, 0.0];
        let y = vec![1.0, 1.0];
        let dy = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let result = conn.covariant_derivative(&p, &x, &y, &dy);
        assert!((result[0] - 1.0).abs() < 1e-12);
        assert!(result[1].abs() < 1e-12);
    }

    #[test]
    fn straight_line_curve_endpoints() {
        let p0 = Point::new(vec![0.0, 0.0]);
        let p1 = Point::new(vec![1.0, 1.0]);
        let curve = straight_line_curve(&p0, &p1, 10);
        let start = curve.evaluate(0.0).unwrap();
        let end = curve.evaluate(1.0).unwrap();
        assert!((start.coords[0]).abs() < 1e-10);
        assert!((end.coords[0] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn parallel_transport_flat_connection() {
        let conn = AffineConnection::new("flat", |_, _, _, _| 0.0);
        let p0 = Point::new(vec![0.0, 0.0]);
        let p1 = Point::new(vec![1.0, 1.0]);
        let curve = straight_line_curve(&p0, &p1, 20);
        let v = TangentVector::at(p0, vec![1.0, 0.0]);
        let transported = conn.parallel_transport(&v, &curve, 100);
        assert!((transported.components[0] - 1.0).abs() < 1e-6);
        assert!((transported.components[1]).abs() < 1e-6);
    }

    #[test]
    fn christoffel_at_returns_correct_shape() {
        let conn = AffineConnection::new("test", |_, _, _, _| 0.0);
        let p = Point::origin(3);
        let gamma = conn.christoffel_at(&p);
        assert_eq!(gamma.len(), 3);
        assert_eq!(gamma[0].len(), 3);
        assert_eq!(gamma[0][0].len(), 3);
    }
}
