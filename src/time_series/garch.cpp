#include "openrisk/time_series/garch.hpp"
#include "openrisk/core/lbfgs.hpp"

namespace openrisk::time_series {

template <core::FloatingPoint T>
GarchParams<T> calibrate_garch(const Eigen::VectorX<T>& returns) {
    core::LBFGSOptimizer<T> opt;
    
    auto objective = [&](const Eigen::VectorX<T>& theta) -> T {
        // theta = [omega, alpha, beta]
        if (theta(0) <= 0 || theta(1) < 0 || theta(2) < 0 || (theta(1) + theta(2) >= 1.0)) {
            return 1e10; // 惩罚无效参
        }
        GarchParams<T> params{theta(0), theta(1), theta(2)};
        return GarchModel<T>::log_likelihood(returns, params);
    };

    Eigen::VectorX<T> x0(3);
    x0 << returns.variance() * 0.1, 0.1, 0.8;

    auto result = opt.minimize(objective, x0, Eigen::VectorX<T>(), Eigen::VectorX<T>());
    return {result.x_best(0), result.x_best(1), result.x_best(2)};
}

} // namespace openrisk::time_series
