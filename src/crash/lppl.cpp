#include "openrisk/crash/lppl.hpp"
#include "openrisk/core/lbfgs.hpp"
#include <algorithm>
#include <limits>
#include <iostream>

namespace openrisk::crash {

template <core::FloatingPoint T>
LPPLParams<T> LPPLCalibrator<T>::calibrate(const Eigen::VectorX<T>& t_series, 
                                           const Eigen::VectorX<T>& log_p_series,
                                           T t_last) {
    core::LBFGSOptimizer<T> opt;
    
    auto objective = [&](const Eigen::VectorX<T>& theta) -> T {
        if (theta.size() < 7) return static_cast<T>(1e10);
        
        LPPLParams<T> p{
            theta(0), theta(1), theta(2), theta(3), theta(4), theta(5), theta(6)
        };

        T sse = LPPLModel<T>::cost_function(t_series, log_p_series, p);
        
        T penalty = 0.0;
        const T lambda = 1e6;
        if (p.m <= 0.0 || p.m >= 1.0) penalty += lambda * std::pow(p.m - 0.5, 2);
        if (p.tc <= t_last) penalty += lambda * std::pow(t_last - p.tc + 1.0, 2);

        return sse + penalty;
    };

    Eigen::VectorX<T> best_x(7);
    best_x << log_p_series.mean(), -0.1, 0.01, t_last + 30.0, 0.5, 8.0, 0.0;
    
    T min_loss = std::numeric_limits<T>::max();

    std::vector<T> m_seeds = {0.5};
    std::vector<T> omega_seeds = {8.0};

    for (auto m_s : m_seeds) {
        for (auto o_s : omega_seeds) {
            Eigen::VectorX<T> x0(7);
            x0 << log_p_series.mean(), -0.1, 0.01, t_last + 30.0, m_s, o_s, 0.0;
            
            auto result = opt.minimize(objective, x0, Eigen::VectorX<T>(), Eigen::VectorX<T>());
            
            if (result.x_best.size() == 7 && result.min_value < min_loss) {
                min_loss = result.min_value;
                best_x = result.x_best;
            }
        }
    }

    return {best_x(0), best_x(1), best_x(2), best_x(3), best_x(4), best_x(5), best_x(6)};
}

template class LPPLCalibrator<double>;

} // namespace openrisk::crash
