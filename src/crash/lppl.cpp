#include "openrisk/crash/lppl.hpp"
#include "openrisk/core/lbfgs.hpp"
#include <algorithm>
#include <limits>

namespace openrisk::crash {

template <core::FloatingPoint T>
class LPPLCalibrator {
public:
    /**
     * @brief LPPL 自动校准函数
     * @param t_series 时间轴序列
     * @param log_p_series 对数价格序列
     * @param t_last 当前最后观测时间
     */
    static LPPLParams<T> calibrate(const Eigen::VectorX<T>& t_series, 
                                   const Eigen::VectorX<T>& log_p_series,
                                   T t_last) {
        core::LBFGSOptimizer<T> opt;
        
        auto objective = [&](const Eigen::VectorX<T>& theta) -> T {
            LPPLParams<T> p{
                theta(0), // A
                theta(1), // B
                theta(2), // C
                theta(3), // tc
                theta(4), // m
                theta(5), // omega
                theta(6)  // phi
            };

            T sse = LPPLModel<T>::cost_function(t_series, log_p_series, p);

            T penalty = 0.0;
            const T lambda = 1e6; // 惩罚强度

            if (p.m <= 0.0 || p.m >= 1.0) penalty += lambda * std::pow(p.m - 0.5, 2);
            if (p.tc <= t_last) penalty += lambda * std::pow(t_last - p.tc + 1.0, 2);
            if (p.B >= 0) penalty += lambda * std::pow(p.B, 2); // 泡沫增长要求 B < 0
            if (std::abs(p.C) > std::abs(p.B)) penalty += lambda * std::pow(p.C, 2); // 震荡不应超过主趋势

            return sse + penalty;
        };

        // LPPL 对初值极其敏感，这里我们生成一个较优的种子
        Eigen::VectorX<T> best_x;
        T min_loss = std::numeric_limits<T>::max();

        std::vector<T> m_seeds = {0.3, 0.5, 0.7};
        std::vector<T> omega_seeds = {4.0, 8.0, 12.0};

        for (auto m_s : m_seeds) {
            for (auto o_s : omega_seeds) {
                Eigen::VectorX<T> x0(7);
                x0 << log_p_series.mean(), -0.1, 0.01, t_last + 30.0, m_s, o_s, 0.0;
                
                auto result = opt.minimize(objective, x0, Eigen::VectorX<T>(), Eigen::VectorX<T>());
                
                if (result.min_value < min_loss) {
                    min_loss = result.min_value;
                    best_x = result.x_best;
                }
            }
        }

        return {best_x(0), best_x(1), best_x(2), best_x(3), best_x(4), best_x(5), best_x(6)};
    }
};

// 显式实例化
template class LPPLCalibrator<double>;

} // namespace openrisk::crash
