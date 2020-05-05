#include <Eigen/Eigenvalues>
#include "opoly1d.h"

using namespace UncertainSCI;
using namespace Eigen;

Matrix3D OPoly1D::eval_driver(const XType& x, const IntList& n, int d, const Matrix2D& ab)
{
  ERROR_NOT_IMPLEMENTED
}

Matrix2D OPoly1D::ratio_driver(const XType& x, int n, int d, const Matrix2D& ab)
{
  ERROR_NOT_IMPLEMENTED
}

Matrix2D OPoly1D::s_driver(const XType& x, int n, const Matrix2D& ab)
{
  ERROR_NOT_IMPLEMENTED
}

Matrix2D OPoly1D::jacobi_matrix_driver(const Matrix2D& ab, int N)
{
  Matrix2D ret(N,N);
  ret.fill(0);
  ret.diagonal(-1) = ab.block(1, 1, N-1, 1);
  ret.diagonal() = ab.block(1, 0, N, 1);
  ret.diagonal(1) = ret.diagonal(-1);
  return ret;//np.diag(ab[1:N,1], k=1) + np.diag(ab[1:(N+1),0],k=0) + np.diag(ab[1:N,1], k=-1);
}

std::tuple<Vector1D, Matrix2D> OPoly1D::gauss_quadrature_driver(const Matrix2D& ab, int N)
{
  //from numpy.linalg import eigh
  //print(ab);
  auto m = jacobi_matrix_driver(ab, N);
  //print(m);

  EigenSolver<Matrix2D> es;
  es.compute(m, true);

  auto lamb = es.eigenvalues();
  //print(lamb);
  auto v = es.eigenvectors();
  return {lamb.real(), (std::pow(ab(0,1), 2) * v.col(0).array().square()).matrix().real()};
}

Matrix2D OPoly1D::markov_stiltjies(const XType& u, int n, const Vector1D& a, const Vector1D& b, const Vector1D& supp)
{
  ERROR_NOT_IMPLEMENTED
}

Vector1D OPoly1D::idistinv_driver(const XType& u, const XType& n, Function1D primitive,
  const Vector1D& a, const Vector1D& b, const Vector1D& supp)
{
  ERROR_NOT_IMPLEMENTED
}

Matrix2D OrthogonalPolynomialBasis1D::recurrence(int N)
{
  ERROR_NOT_IMPLEMENTED
}

Matrix3D OrthogonalPolynomialBasis1D::eval(const XType& x, int n, int d)
{
  ERROR_NOT_IMPLEMENTED
}

Matrix2D OrthogonalPolynomialBasis1D::jacobi_matrix_driver(const Matrix2D& ab, int N)
{
  ERROR_NOT_IMPLEMENTED
}

Matrix2D OrthogonalPolynomialBasis1D::jacobi_matrix(int N)
{
  ERROR_NOT_IMPLEMENTED
}

Matrix2D OrthogonalPolynomialBasis1D::apply_jacobi_matrix(const Vector1D& v)
{
  ERROR_NOT_IMPLEMENTED
}

std::tuple<double, Matrix2D> OrthogonalPolynomialBasis1D::gauss_quadrature(int N)
{
  ERROR_NOT_IMPLEMENTED
}

std::tuple<double, Matrix2D> OrthogonalPolynomialBasis1D::gauss_radau_quadrature(int N, double anchor)
{
  ERROR_NOT_IMPLEMENTED
}

Vector1D OrthogonalPolynomialBasis1D::leading_coefficient(int N)
{
  ERROR_NOT_IMPLEMENTED
}

Matrix2D OrthogonalPolynomialBasis1D::canonical_connection(int N)
{
  ERROR_NOT_IMPLEMENTED
}

Matrix2D OrthogonalPolynomialBasis1D::canonical_connection_inverse(int N)
{
  ERROR_NOT_IMPLEMENTED
}

Matrix2D OrthogonalPolynomialBasis1D::tuple_product_generator(const Vector1D& IC, const Matrix2D& ab)
{
  ERROR_NOT_IMPLEMENTED
}

Matrix2D OrthogonalPolynomialBasis1D::tuple_product(int N, const Vector1D& alpha)
{
  ERROR_NOT_IMPLEMENTED
}

Matrix2D OrthogonalPolynomialBasis1D::derivative_expansion(int N, int d)
{
  ERROR_NOT_IMPLEMENTED
}

Matrix2D OrthogonalPolynomialBasis1D::r_eval(const XType& x, int n, int d)
{
  ERROR_NOT_IMPLEMENTED
}

Matrix2D OrthogonalPolynomialBasis1D::s_eval(const XType& x, int n)
{
  ERROR_NOT_IMPLEMENTED
}

Matrix2D OrthogonalPolynomialBasis1D::qpoly1d_eval(const XType& x, int n, int d)
{
  ERROR_NOT_IMPLEMENTED
}

Matrix2D OrthogonalPolynomialBasis1D::christoffel_function(const XType& x, int k)
{
  ERROR_NOT_IMPLEMENTED
}

Matrix2D OrthogonalPolynomialBasis1D::recurrence_quad_mod_jacobi(const Matrix2D& ab, int N, const XType& z0)
{
  ERROR_NOT_IMPLEMENTED
}

Matrix2D OrthogonalPolynomialBasis1D::recurrence_lin_mod_jacobi(const Matrix2D& ab, int N, const XType& y0)
{
  ERROR_NOT_IMPLEMENTED
}
