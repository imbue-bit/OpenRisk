#include "openrisk/time_series/garch.hpp"
#include "openrisk/core/lbfgs.hpp"
#include "openrisk/core/stats.hpp"
#include <iostream>

namespace openrisk::time_series {

template <core::FloatingPoint T>
GarchParams<T> calibrate_garch(const Eigen::VectorX<T>& returns) {
    if (returns.size() < 10) return {0.0001, 0.05, 0.9};

    core::LBFGSOptimizer<T> opt;
    auto objective = [&](const Eigen::VectorX<T>& theta) -> T {
        if (theta.size() < 3) return static_cast<T>(1e10);
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

    if (result.x_best.size() < 3) {
        std::cerr << "Warning: GARCH optimization failed, using initial guess.\n";
        return {x0(0), x0(1), x0(2)};
    }
    
    return {result.x_best(0), result.x_best(1), result.x_best(2)};
}

template GarchParams<double> calibrate_garch<double>(const Eigen::VectorX<double>& returns);

} // namespace openrisk::time_series
