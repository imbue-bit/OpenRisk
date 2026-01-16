#pragma once
#include "concepts.hpp"
#include <functional>

namespace openrisk::core {

/**
 * @brief 优化目标函数定义
 * 输入参数向量，返回标量 Loss
 */
template <FloatingPoint T = double>
using ObjectiveFunc = std::function<T(const Eigen::VectorX<T>&)>;

template <FloatingPoint T = double>
struct OptimizationResult {
    Eigen::VectorX<T> x_best;
    T min_value;
    int iterations;
    bool converged;
};

/**
 * @brief 优化器接口基类
 */
template <FloatingPoint T = double>
class Optimizer {
public:
    virtual ~Optimizer() = default;
    virtual OptimizationResult<T> minimize(
        ObjectiveFunc<T> func, 
        const Eigen::VectorX<T>& x0,
        const Eigen::VectorX<T>& lower_bounds,
        const Eigen::VectorX<T>& upper_bounds) = 0;
};

} // namespace openrisk::core
