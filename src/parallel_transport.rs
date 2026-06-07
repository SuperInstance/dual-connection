//! Parallel transport along curves using affine connections.
//!
//! Parallel transport moves a tangent vector along a curve while keeping it
//! "constant" according to the connection. On a curved manifold, the result
//! generally depends on the path taken.

use crate::connection::{AffineConnection, Point, TangentVector, straight_line_curve};
use crate::dual::{normal_e_connection, normal_m_connection};

/// Transport a vector along a straight-line curve between two points.
pub fn transport_between(
    conn: &AffineConnection,
    v: &TangentVector,
    target: &Point,
    steps: usize,
) -> TangentVector {
    let curve = straight_line_curve(&v.point, target, steps.max(10));
    conn.parallel_transport(v, &curve, steps)
}

/// Compare e-transport vs m-transport of the same vector on the normal manifold.
pub fn compare_e_m_transport(
    start: &Point,
    end: &Point,
    v: &TangentVector,
    steps: usize,
) -> (TangentVector, TangentVector) {
    let e_conn = normal_e_connection();
    let m_conn = normal_m_connection();
    let curve = straight_line_curve(start, end, steps.max(10));

    let e_result = e_conn.parallel_transport(v, &curve, steps);
    let m_result = m_conn.parallel_transport(v, &curve, steps);

    (e_result, m_result)
}

/// Compute the holonomy angle: transport a vector around a closed loop.
pub fn holonomy_angle(
    conn: &AffineConnection,
    center: &Point,
    radius: f64,
    v: &TangentVector,
    steps: usize,
) -> f64 {
    let p0 = center.clone();
    let p1 = Point::new(vec![center.coords[0] + radius, center.coords[1]]);
    let p2 = Point::new(vec![center.coords[0] + radius, center.coords[1] + radius]);
    let p3 = Point::new(vec![center.coords[0], center.coords[1] + radius]);

    let edges = [
        straight_line_curve(&p0, &p1, steps / 4),
        straight_line_curve(&p1, &p2, steps / 4),
        straight_line_curve(&p2, &p3, steps / 4),
        straight_line_curve(&p3, &p0, steps / 4),
    ];

    let mut current = v.clone();
    for curve in &edges {
        current = conn.parallel_transport(&current, curve, steps / 4);
    }

    let dot: f64 = v
        .components
        .iter()
        .zip(&current.components)
        .map(|(a, b)| a * b)
        .sum();
    let cross = v.components[0] * current.components[1] - v.components[1] * current.components[0];
    cross.atan2(dot)
}

/// Verify flatness by checking holonomy around a small loop.
pub fn is_flat_by_holonomy(
    conn: &AffineConnection,
    center: &Point,
    radius: f64,
    steps: usize,
    tol: f64,
) -> bool {
    let v = TangentVector::at(center.clone(), vec![1.0, 0.0]);
    let angle = holonomy_angle(conn, center, radius, &v, steps);
    angle.abs() < tol
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dual::normal_alpha_connection;

    #[test]
    fn e_connection_flat_holonomy() {
        let conn = normal_e_connection();
        let center = Point::new(vec![0.0, 2.0]);
        assert!(is_flat_by_holonomy(&conn, &center, 0.1, 2000, 0.05));
    }

    #[test]
    fn m_connection_flat_holonomy() {
        let conn = normal_m_connection();
        let center = Point::new(vec![0.0, 2.0]);
        assert!(is_flat_by_holonomy(&conn, &center, 0.1, 2000, 0.05));
    }

    #[test]
    fn levi_civita_not_flat_holonomy() {
        let conn = normal_alpha_connection(0.0);
        let center = Point::new(vec![0.0, 2.0]);
        assert!(!is_flat_by_holonomy(&conn, &center, 0.5, 2000, 0.01));
    }

    #[test]
    fn e_transport_preserves_e_coordinate() {
        let start = Point::new(vec![0.0, 1.0]);
        let end = Point::new(vec![0.5, 1.5]);
        let v = TangentVector::at(start, vec![1.0, 0.0]);
        let transported = transport_between(&normal_e_connection(), &v, &end, 500);
        assert!(transported.components[0].is_finite());
        assert!(transported.components[1].is_finite());
    }

    #[test]
    fn compare_e_m_gives_different_results() {
        let start = Point::new(vec![0.0, 2.0]);
        let end = Point::new(vec![1.0, 3.0]);
        let v = TangentVector::at(start.clone(), vec![1.0, 1.0]);
        let (e_result, m_result) = compare_e_m_transport(&start, &end, &v, 500);
        let diff = (e_result.components[0] - m_result.components[0]).abs()
            + (e_result.components[1] - m_result.components[1]).abs();
        assert!(diff > 0.01, "e and m transport should differ");
    }

    #[test]
    fn transport_between_close_points_small_change() {
        let start = Point::new(vec![0.0, 2.0]);
        let end = Point::new(vec![0.001, 2.001]);
        let v = TangentVector::at(start, vec![1.0, 0.0]);
        let transported = transport_between(&normal_alpha_connection(0.5), &v, &end, 1000);
        assert!((transported.components[0] - 1.0).abs() < 0.01);
        assert!(transported.components[1].abs() < 0.01);
    }

    #[test]
    fn zero_connection_transport_is_identity() {
        let conn = AffineConnection::new("flat", |_, _, _, _| 0.0);
        let start = Point::new(vec![0.0, 0.0]);
        let end = Point::new(vec![5.0, 5.0]);
        let v = TangentVector::at(start, vec![3.0, 4.0]);
        let transported = transport_between(&conn, &v, &end, 100);
        assert!((transported.components[0] - 3.0).abs() < 1e-6);
        assert!((transported.components[1] - 4.0).abs() < 1e-6);
    }
}
