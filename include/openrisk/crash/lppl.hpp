#pragma once
#include "../core/concepts.hpp"
#include <cmath>

namespace openrisk::crash {

template <core::FloatingPoint T = double>
struct LPPLParams {
    T A;     // 崩盘时的期望价格水平
    T B;     // 泡沫增长的振幅
    T C;     // 震荡幅度
    T tc;    // 崩盘点
    T m;     // 幂律指数
    T omega; // 震荡频率
    T phi;   // 相位偏移
};

template <core::FloatingPoint T = double>
class LPPLModel {
public:
    /**
     * @brief LPPL：log(P(t)) = A + B(tc-t)^m + C(tc-t)^m * cos(omega * log(tc-t) + phi)
     */
    static T compute(T t, const LPPLParams<T>& p) {
        T dt = p.tc - t;
        if (dt <= 0) return p.A; // 已过临界点

        T dt_m = std::pow(dt, p.m);
        T log_dt = std::log(dt);
        T oscillating_term = p.C * dt_m * std::cos(p.omega * log_dt + p.phi);
        
        return p.A + p.B * dt_m + oscillating_term;
    }

    /**
     * @brief 计算拟合残差平方和
     */
    static T cost_function(const Eigen::VectorX<T>& t_series, 
                           const Eigen::VectorX<T>& log_p_series, 
                           const LPPLParams<T>& p) 
    {
        T sse = 0.0;
        for (int i = 0; i < t_series.size(); ++i) {
            T pred = compute(t_series(i), p);
            sse += std::pow(pred - log_p_series(i), 2);
        }
        return sse;
    }

    /**
     * @brief 判定泡沫强度指标 (DSLPPL Index)
     */
    static bool is_bubble_present(const LPPLParams<T>& p) {
        return (p.m > 0 && p.m < 1 && p.B < 0);
    }
};

} // namespace openrisk::crash
