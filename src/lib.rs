//! # dual-connection
//!
//! Dual affine connections (e-connection, m-connection) on statistical manifolds.
//!
//! This crate provides the core geometric structures of **information geometry**:
//! affine connections, their duals, curvature, torsion, parallel transport,
//! and the divergences that arise from dually flat structures.
//!
//! ## Example
//!
//! ```
//! use dual_connection::dual::{normal_e_connection, normal_m_connection};
//! use dual_connection::connection::Point;
//!
//! let p = Point::new(vec![0.0, 1.0]); // N(μ=0, σ=1)
//! let e_conn = normal_e_connection();
//! let m_conn = normal_m_connection();
//!
//! let gamma_e = e_conn.christoffel_at(&p);
//! let gamma_m = m_conn.christoffel_at(&p);
//! ```

pub mod connection;
pub mod curvature;
pub mod divergence;
pub mod dual;
pub mod parallel_transport;
pub mod torsion;

pub use connection::{AffineConnection, Curve, Point, TangentVector};
pub use curvature::CurvatureTensor;
pub use divergence::{BregmanDivergence, PythagoreanResult};
pub use dual::Metric;
pub use torsion::TorsionTensor;
