#include "families.h"

using namespace UncertainSCI;

Matrix2D Families::jacobi_recurrence_values(int N, double alpha, double beta)
{
  // """
  // Returns the first N+1 recurrence coefficient pairs for the (alpha, beta)
  // Jacobi family
  // """
  if (N < 1)
  {
    auto ab = np::ones(1, 2);
    ab(0,0) = 0;
    ab(0,1) = std::exp( (alpha + beta + 1.) * std::log(2.) +
                  std::lgamma(alpha + 1.) + std::lgamma(beta + 1.) -
                  std::lgamma(alpha + beta + 2.)
                );
    return ab;
  }

  Matrix2D ab(N+1, 2);
  ab.col(0).fill(std::pow(beta, 2.) - std::pow(alpha, 2.));
  ab.col(1).fill(1);

  //# Special cases
  ab(0,0) = 0.;
  ab(1,0) = (beta - alpha) / (alpha + beta + 2.);
  ab(0,1) = std::exp( (alpha + beta + 1.) * std::log(2.) +
                    std::lgamma(alpha + 1.) + std::lgamma(beta + 1.) -
                    std::lgamma(alpha + beta + 2.)
                  );

  ab(1,1) = 4. * (alpha + 1.) * (beta + 1.) / (
                 std::pow((alpha + beta + 2.), 2) * (alpha + beta + 3.) );

  if (N > 1)
  {
    ab(1,1) = 4. * (alpha + 1.) * (beta + 1.) / (
               std::pow((alpha + beta + 2.), 2) * (alpha + beta + 3.) );

    ab(2,0) /= (2. + alpha + beta) * (4. + alpha + beta);
    auto inds = 2;
    ab(2,1) = 4 * inds * (inds + alpha) * (inds + beta) * (inds + alpha + beta);
    ab(2,1) /= std::pow((2. * inds + alpha + beta), 2) * (2. * inds + alpha + beta + 1.) * (2. * inds + alpha + beta - 1);
  }
#if 0
  if (N > 2)
  {
    auto inds = np::arange(2., N+1);
    ab(3:,0) /= (2. * inds[:-1] + alpha + beta) * (2 * inds[:-1] + alpha + beta + 2.);
    ab(2:,1) = 4 * inds * (inds + alpha) * (inds + beta) * (inds + alpha + beta);
    ab(2:,1) /= (2. * inds + alpha + beta)**2 * (2. * inds + alpha + beta + 1.) * (2. * inds + alpha + beta - 1);
  }
#endif
  ab.col(1) = ab.col(1).cwiseSqrt();

  return ab;
}
