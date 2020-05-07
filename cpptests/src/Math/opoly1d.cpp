#include <Eigen/Eigenvalues>
#include "opoly1d.h"

using namespace UncertainSCI;
using namespace Eigen;

Matrix3D OPoly1D::eval_driver(const XType& x, const IntList& n, int d, const Matrix2D& ab)
{
  auto nmax = static_cast<int>(*std::max_element(EIGEN_BEGIN_END(n)));

  auto xf = std::get<Vector1D>(x);
  auto p = Matrix2D( xf.rows(), nmax+1);
  p.fill(0);

  p.col(0) = p.col(0).array() + 1/ab(0,1);

  if (nmax > 0)
    p.col(1) = p.col(1).array() + 1/ab(1,1) * ( (xf.array() - ab(1,0)) * p.col(0).array() );

  for (int j : np::range(2, nmax+1))
    p.col(j) = 1/ab(j,1) * ( (xf.array() - ab(j,0)) * p.col(j-1).array() - ab(j-1,1)*p.col(j-2).array() );

  std::vector<int> ds;
  //TODO: support other dynamic inputs
  if (true) //type(d) == int)
  {
    if (d == 0)
      return {p};//[:,n.flatten()]
    else
      ds = { d };
  }

//TODO: no derivatives needed yet
#if 0
  auto preturn = np.zeros([p.shape[0], n.size, len(d)]);

  // # Parse the list d to find which indices contain which
  // # derivative orders

  for (i in [i for i,val in enumerate(d) if val==0])
    preturn[:,:,i] = p[:,n.flatten()];

  for qd in range(1, max(d)+1)
  {
    auto pd = np.zeros(p.shape)

    for (qn in range(qd,nmax+1))
    {
      if (qn == qd)
        pd[:,qn] = np.exp( sp.gammaln(qd+1) - np.sum( np.log( ab[:(qd+1),1] ) ) );
      else
        pd[:,qn] = 1/ab[qn,1] * ( ( xf - ab[qn,0] ) * pd[:,qn-1] - ab[qn-1,1] * pd[:,qn-2] + qd*p[:,qn-1] );
    }

    for (i in [i for i,val in enumerate(d) if val==qd])
      preturn[:,:,i] = pd[:,n.flatten()];

    p = pd;
  }
  if (len(d) == 1)
    return preturn.squeeze(axis=2);
  else
    return preturn;
#endif
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
  Matrix2D diag(N,N);
  diag.fill(0);
  diag.diagonal(-1) = ab.block(1, 1, N-1, 1);
  diag.diagonal() = ab.block(1, 0, N, 1);
  diag.diagonal(1) = diag.diagonal(-1);
  return diag;
}

std::tuple<Vector1D, Matrix2D> OPoly1D::gauss_quadrature_driver(const Matrix2D& ab, int N)
{
  auto m = jacobi_matrix_driver(ab, N);

  EigenSolver<Matrix2D> es;
  es.compute(m, true);

  auto lamb = es.eigenvalues().real().eval();
  std::sort(EIGEN_BEGIN_END(lamb));
  auto v = es.eigenvectors();
  return {lamb, (std::pow(ab(0,1), 2) * v.col(0).array().square()).matrix().real()};
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
