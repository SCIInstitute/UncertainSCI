#include "opoly1d.h"

using namespace UncertainSCI;
using namespace TODO_REPLACE_WITH_EIGEN;


Matrix3D OPoly1D::eval_driver(const XType& x, const IntList& n, int d, const Matrix2D& ab)
{
  throw "TODO";
}

Matrix2D OPoly1D::ratio_driver(const XType& x, int n, int d, const Matrix2D& ab)
{
  throw "TODO";
}

Matrix2D OPoly1D::s_driver(const XType& x, int n, const Matrix2D& ab)
{
  throw "TODO";
}

Matrix2D OPoly1D::jacobi_matrix_driver(const Matrix2D& ab, int N)
{
  throw "TODO";
}

std::tuple<double, Matrix2D> OPoly1D::gauss_quadrature_driver(const Matrix2D& ab, int N)
{
  throw "TODO";
}

Matrix2D OPoly1D::markov_stiltjies(const XType& u, int n, const Vector1D& a, const Vector1D& b, const Vector1D& supp)
{
  throw "TODO";
}

Vector1D OPoly1D::idistinv_driver(const XType& u, const XType& n, Function1D primitive,
  const Vector1D& a, const Vector1D& b, const Vector1D& supp)
{
  throw "TODO";
}

Matrix2D OrthogonalPolynomialBasis1D::recurrence(int N)
{
  throw "TODO";
}

Matrix3D OrthogonalPolynomialBasis1D::eval(const XType& x, int n, int d)
{
  throw "TODO";
}

Matrix2D OrthogonalPolynomialBasis1D::jacobi_matrix_driver(const Matrix2D& ab, int N)
{
  throw "TODO";
}

Matrix2D OrthogonalPolynomialBasis1D::jacobi_matrix(int N)
{
  throw "TODO";
}

Matrix2D OrthogonalPolynomialBasis1D::apply_jacobi_matrix(const Vector1D& v)
{
  throw "TODO";
}

std::tuple<double, Matrix2D> OrthogonalPolynomialBasis1D::gauss_quadrature(int N)
{
  throw "TODO";
}

std::tuple<double, Matrix2D> OrthogonalPolynomialBasis1D::gauss_radau_quadrature(int N, double anchor)
{
  throw "TODO";
}

Vector1D OrthogonalPolynomialBasis1D::leading_coefficient(int N)
{
  throw "TODO";
}

Matrix2D OrthogonalPolynomialBasis1D::canonical_connection(int N)
{
  throw "TODO";
}

Matrix2D OrthogonalPolynomialBasis1D::canonical_connection_inverse(int N)
{
  throw "TODO";
}

Matrix2D OrthogonalPolynomialBasis1D::tuple_product_generator(const Vector1D& IC, const Matrix2D& ab)
{
  throw "TODO";
}

Matrix2D OrthogonalPolynomialBasis1D::tuple_product(int N, const Vector1D& alpha)
{
  throw "TODO";
}

Matrix2D OrthogonalPolynomialBasis1D::derivative_expansion(int N, int d)
{
  throw "TODO";
}

Matrix2D OrthogonalPolynomialBasis1D::r_eval(const XType& x, int n, int d)
{
  throw "TODO";
}

Matrix2D OrthogonalPolynomialBasis1D::s_eval(const XType& x, int n)
{
  throw "TODO";
}

Matrix2D OrthogonalPolynomialBasis1D::qpoly1d_eval(const XType& x, int n, int d)
{
  throw "TODO";
}

Matrix2D OrthogonalPolynomialBasis1D::christoffel_function(const XType& x, int k)
{
  throw "TODO";
}

Matrix2D OrthogonalPolynomialBasis1D::recurrence_quad_mod_jacobi(const Matrix2D& ab, int N, const XType& z0)
{
  throw "TODO";
}

Matrix2D OrthogonalPolynomialBasis1D::recurrence_lin_mod_jacobi(const Matrix2D& ab, int N, const XType& y0)
{
  throw "TODO";
}
