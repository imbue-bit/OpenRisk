#pragma once
#include "../core/concepts.hpp"
#include <vector>

namespace openrisk::time_series {

template <core::FloatingPoint T = double>
struct GarchParams {
    T omega; // 常数
    T alpha; // ARCH
    T beta;  // GARCH
};

template <core::FloatingPoint T = double>
class GarchModel {
public:
    /**
     * @brief 波动率过滤：根据给定参数计算时序波动率 sigma_t
     * @param returns 收益率序列 (已去均值)
     * @param params 模型参数
     * @return 预测的波动率序列 (sigma^2)
     */
    static Eigen::VectorX<T> filter(const Eigen::VectorX<T>& returns, const GarchParams<T>& params) {
        const std::size_t n = returns.size();
        Eigen::VectorX<T> sigmas_sq(n);
        
        T initial_variance = (returns.array().square().sum()) / static_cast<T>(n);
        sigmas_sq(0) = initial_variance;

        for (std::size_t t = 1; t < n; ++t) {
            sigmas_sq(t) = params.omega + 
                           params.alpha * std::pow(returns(t-1), 2) + 
                           params.beta * sigmas_sq(t-1);
        }
        return sigmas_sq;
    }

    /**
     * @brief 计算负对数似然
     * 用于优化器寻找最优参数
     */
    static T log_likelihood(const Eigen::VectorX<T>& returns, const GarchParams<T>& params) {
        const std::size_t n = returns.size();
        Eigen::VectorX<T> sigmas_sq = filter(returns, params);
        
        T log_lik = 0.0;
        const T pi = std::numbers::pi_v<T>;

        for (std::size_t t = 0; t < n; ++t) {
            // 假设正态分布：-0.5 * (log(2*pi) + log(sigma^2) + r^2/sigma^2)
            log_lik += -0.5 * (std::log(2 * pi) + std::log(sigmas_sq(t)) + std::pow(returns(t), 2) / sigmas_sq(t));
        }
        return -log_lik;
    }
};
template <core::FloatingPoint T = double>
GarchParams<T> calibrate_garch(const Eigen::VectorX<T>& returns);
} // namespace openrisk::time_series
