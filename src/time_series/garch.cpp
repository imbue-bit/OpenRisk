#include "openrisk/time_series/garch.hpp"
#include "openrisk/core/lbfgs.hpp"
#include "openrisk/core/stats.hpp"
#include <iostream>

namespace openrisk::time_series {

template <core::FloatingPoint T>
GarchParams<T> calibrate_garch(const Eigen::VectorX<T>& returns) {
    if (returns.size() < 5) {
        return {static_cast<T>(0.0001), static_cast<T>(0.05), static_cast<T>(0.9)};
    }

    core::LBFGSOptimizer<T> opt;
    
    auto objective = [&](const Eigen::VectorX<T>& theta) -> T {
        if (theta.size() < 3) return static_cast<T>(1e10);

        // omega > 0, alpha, beta >= 0, alpha + beta < 1
        if (theta(0) <= 0 || theta(1) < 0 || theta(2) < 0 || (theta(1) + theta(2) >= 1.0)) {
            return static_cast<T>(1e10); 
        }
        
        GarchParams<T> params{theta(0), theta(1), theta(2)};
        return GarchModel<T>::log_likelihood(returns, params);
    };

    Eigen::VectorX<T> x0(3);
    T var = core::Statistics<T>::variance(returns);
    // omega较小，alpha=0.1, beta=0.8
    x0 << var * static_cast<T>(0.1), static_cast<T>(0.1), static_cast<T>(0.8);

    auto result = opt.minimize(objective, x0, Eigen::VectorX<T>(), Eigen::VectorX<T>());

    // 如果优化器由于某种原因返回了空向量，或者 size 不对，直接返回初始值 x0
    if (result.x_best.size() < 3) {
        return {x0(0), x0(1), x0(2)};
    }
    
    return {result.x_best(0), result.x_best(1), result.x_best(2)};
}

// 显式实例化
template GarchParams<double> calibrate_garch<double>(const Eigen::VectorX<double>& returns);

} // namespace openrisk::time_series
