#pragma once
#include "openrisk/core/concepts.hpp"

namespace openrisk::factor {

template <core::FloatingPoint T = double>
class RiskAttribution {
public:
    /**
     * @brief 风险分解：计算因子风险和残差风险
     * @param weights 资产权重 (N x 1)
     * @param beta 因子暴露矩阵 (N x K)
     * @param factor_cov 因子协方差矩阵 (K x K)
     * @param specific_var 特异性风险向量 (N x 1)
     */
    static T total_variance(const Eigen::VectorX<T>& weights,
                            const Eigen::MatrixX<T>& beta,
                            const Eigen::MatrixX<T>& factor_cov,
                            const Eigen::VectorX<T>& specific_var) {
        Eigen::VectorX<T> exposure = beta.transpose() * weights;
        T factor_risk = exposure.transpose() * factor_cov * exposure;

        T idiosyncratic_risk = (weights.array().square() * specific_var.array()).sum();

        return factor_risk + idiosyncratic_risk;
    }

    /**
     * @brief 计算MCTR
     */
    static Eigen::VectorX<T> marginal_contribution_to_risk(
        const Eigen::VectorX<T>& weights,
        const Eigen::MatrixX<T>& full_covariance) {
        
        T portfolio_vol = std::sqrt(weights.transpose() * full_covariance * weights);
        return (full_covariance * weights) / portfolio_vol;
    }
};

} // namespace openrisk::factor
