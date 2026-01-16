#pragma once
#include "../core/stats.hpp"
#include <algorithm>

namespace openrisk::tail {

template <core::FloatingPoint T = double>
class RiskMetrics {
public:
    /**
     * @brief 历史模拟法 Value at Risk
     */
    static T historical_var(Eigen::VectorX<T> returns, T confidence_level = 0.95) {
        std::sort(returns.data(), returns.data() + returns.size());
        std::size_t index = static_cast<std::size_t>((1.0 - confidence_level) * returns.size());
        return -returns(index);
    }

    /**
     * @brief 参数法 Value at Risk (正态分布假设)
     */
    static T parametric_var(const Eigen::VectorX<T>& returns, T confidence_level = 0.95) {
        T sigma = core::Statistics<T>::standard_deviation(returns);
        T mean = core::Statistics<T>::mean(returns);
        
        T z_score = (confidence_level > 0.97) ? 2.326 : 1.645; 
        return -(mean - z_score * sigma);
    }

    /**
     * @brief Expected Shortfall (CVaR)
     */
    static T expected_shortfall(Eigen::VectorX<T> returns, T confidence_level = 0.95) {
        std::sort(returns.data(), returns.data() + returns.size());
        std::size_t cutoff = static_cast<std::size_t>((1.0 - confidence_level) * returns.size());
        
        T sum = 0.0;
        for (std::size_t i = 0; i < cutoff; ++i) {
            sum += returns(i);
        }
        return -(sum / static_cast<T>(cutoff));
    }
};

} // namespace openrisk::tail
