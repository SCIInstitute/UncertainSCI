#include "opoly1d.h"

using namespace UncertainSCI;

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
  ERROR_NOT_IMPLEMENTED
}

std::tuple<double, Matrix2D> OPoly1D::gauss_quadrature_driver(const Matrix2D& ab, int N)
{
  ERROR_NOT_IMPLEMENTED
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
