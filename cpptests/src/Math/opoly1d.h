#ifndef SCI_UNCERTAINSCI_OPOLY1D_H
#define SCI_UNCERTAINSCI_OPOLY1D_H

#include <variant>
#include <vector>

namespace TODO_REPLACE_WITH_EIGEN
{
  using Vector1D = std::vector<double>;
  using Matrix2D = std::vector<std::vector<double>>;
  using Matrix3D = std::vector<std::vector<std::vector<double>>>;
  using XType = std::variant<int, double, Vector1D>;
}

namespace UncertainSCI
{

  using Function1D = std::function<double(double)>;
  using namespace TODO_REPLACE_WITH_EIGEN;

class OPoly1D
{
public:
  static Matrix3D eval_driver(const XType& x, int n, int d, const Matrix2D& ab);
  static Matrix2D ratio_driver(const XType& x, int n, int d, const Matrix2D& ab);
  static Matrix2D s_driver(const XType& x, int n, const Matrix2D& ab);
  static Matrix2D jacobi_matrix_driver(const Matrix2D& ab, int N);
  static std::tuple<double, Matrix2D> gauss_quadrature_driver(const Matrix2D& ab, int N);
  static Matrix2D markov_stiltjies(const XType& u, int n, const Vector1D& a, const Vector1D& b, const Vector1D& supp);
  static Vector1D idistinv_driver(const XType& u, const XType& n, Function1D primitive,
    const Vector1D& a, const Vector1D& b, const Vector1D& supp);
};

class OrthogonalPolynomialBasis1D
{
public:
  Matrix2D recurrence(int N);
  Matrix3D eval(const XType& x, int n, int d = 0);
  Matrix2D jacobi_matrix_driver(const Matrix2D& ab, int N);
  Matrix2D jacobi_matrix(int N);
  //TODO: raises dimension by 1 generically
  Matrix2D apply_jacobi_matrix(const Vector1D& v);
  std::tuple<double, Matrix2D> gauss_quadrature(int N);
  std::tuple<double, Matrix2D> gauss_radau_quadrature(int N, double anchor = 0.0);
  Vector1D leading_coefficient(int N);
  Matrix2D canonical_connection(int N);
  Matrix2D canonical_connection_inverse(int N);
  Matrix2D tuple_product_generator(const Vector1D& IC, const Matrix2D& ab = {});
  Matrix2D tuple_product(int N, const Vector1D& alpha);
  Matrix2D derivative_expansion(int N, int d);
  Matrix2D r_eval(const XType& x, int n, int d = 0);
  Matrix2D s_eval(const XType& x, int n);
  Matrix2D qpoly1d_eval(const XType& x, int n, int d = 0);
  Matrix2D christoffel_function(const XType& x, int k);
  Matrix2D recurrence_quad_mod_jacobi(const Matrix2D& ab, int N, const XType& z0);
  Matrix2D recurrence_lin_mod_jacobi(const Matrix2D& ab, int N, const XType& y0);
private:
  bool probability_measure {true};
  Matrix2D ab;// = np.zeros([0,2]);
};


}

#endif
