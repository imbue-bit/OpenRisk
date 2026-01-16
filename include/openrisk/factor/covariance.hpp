#pragma once
#include "../core/stats.hpp"

namespace openrisk::factor {

template <core::FloatingPoint T = double>
class CovarianceEstimator {
public:
    /**
     * @brief 基础样本协方差矩阵
     */
    static Eigen::MatrixX<T> sample_covariance(const Eigen::MatrixX<T>& returns) {
        std::size_t n = returns.rows();
        Eigen::MatrixX<T> centered = returns.rowwise() - returns.colwise().mean();
        return (centered.adjoint() * centered) / static_cast<T>(n - 1);
    }

    /**
     * @brief Ledoit-Wolf 收缩估算
     */
    static Eigen::MatrixX<T> ledoit_wolf_shrinkage(const Eigen::MatrixX<T>& returns) {
        const std::size_t n = returns.rows(); // 样本数
        const std::size_t p = returns.cols(); // 资产数
        
        Eigen::MatrixX<T> sample_cov = sample_covariance(returns);
        
        T mean_var = sample_cov.diagonal().mean();
        T sum_corr = 0.0;
        for (std::size_t i = 0; i < p; ++i) {
            for (std::size_t j = i + 1; j < p; ++j) {
                sum_corr += sample_cov(i, j) / std::sqrt(sample_cov(i, i) * sample_cov(j, j));
            }
        }
        T average_corr = (p > 1) ? (2.0 * sum_corr / (p * (p - 1))) : 0.0;
        
        Eigen::MatrixX<T> target = Eigen::MatrixX<T>::Constant(p, p, average_corr);
        for (std::size_t i = 0; i < p; ++i) {
            for (std::size_t j = 0; j < p; ++j) {
                if (i != j) {
                    target(i, j) *= std::sqrt(sample_cov(i, i) * sample_cov(j, j));
                } else {
                    target(i, j) = sample_cov(i, i);
                }
            }
        }

        T delta = estimate_shrinkage_intensity(returns, sample_cov, target);
        
        return (1.0 - delta) * sample_cov + delta * target;
    }

private:
    static T estimate_shrinkage_intensity(const Eigen::MatrixX<T>& returns, 
                                         const Eigen::MatrixX<T>& S, 
                                         const Eigen::MatrixX<T>& F) {
        T p = static_cast<T>(returns.cols());
        T n = static_cast<T>(returns.rows());
        T ratio = p / n;
        return std::clamp(ratio, 0.01, 0.99);
    }
};

} // namespace openrisk::factor
