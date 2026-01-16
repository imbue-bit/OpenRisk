#include "openrisk/time_series/garch.hpp"
#include "openrisk/core/lbfgs.hpp"
#include "openrisk/core/stats.hpp"

namespace openrisk::time_series {

template <core::FloatingPoint T>
GarchParams<T> calibrate_garch(const Eigen::VectorX<T>& returns) {
    core::LBFGSOptimizer<T> opt;
    
    auto objective = [&](const Eigen::VectorX<T>& theta) -> T {
        // theta = [omega, alpha, beta]
        if (theta(0) <= 0 || theta(1) < 0 || theta(2) < 0 || (theta(1) + theta(2) >= 1.0)) {
            return static_cast<T>(1e10); 
        }
        GarchParams<T> params{theta(0), theta(1), theta(2)};
        return GarchModel<T>::log_likelihood(returns, params);
    };

    Eigen::VectorX<T> x0(3);
    T var = core::Statistics<T>::variance(returns);
    x0 << var * static_cast<T>(0.1), static_cast<T>(0.1), static_cast<T>(0.8);

    auto result = opt.minimize(objective, x0, Eigen::VectorX<T>(), Eigen::VectorX<T>());
    return {result.x_best(0), result.x_best(1), result.x_best(2)};
}

template GarchParams<double> calibrate_garch<double>(const Eigen::VectorX<double>& returns);

} // namespace openrisk::time_series
