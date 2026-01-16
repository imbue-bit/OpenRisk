#pragma once
#include "optimization.hpp"
#include <deque>

namespace openrisk::core {

template <FloatingPoint T = double>
class LBFGSOptimizer : public Optimizer<T> {
public:
    struct Config {
        int m = 10;                     // 记忆历史步数
        int max_iter = 500;             // 最大迭代次数
        T g_tol = 1e-6;                 // 梯度收敛阈值
        T step_alpha = 1.0;             // 初始步长
        T wolfe_c1 = 1e-4;              // Wolfe 准则参数 1
        T wolfe_c2 = 0.9;               // Wolfe 准则参数 2
    };

    explicit LBFGSOptimizer(Config config = Config()) : config_(config) {}

    OptimizationResult<T> minimize(
        ObjectiveFunc<T> func,
        const Eigen::VectorX<T>& x0,
        const Eigen::VectorX<T>& lower_bounds,
        const Eigen::VectorX<T>& upper_bounds) override;

private:
    struct StepHistory {
        Eigen::VectorX<T> s; // x_{k+1} - x_k
        Eigen::VectorX<T> y; // g_{k+1} - g_k
        T rho;               // 1 / (y^T * s)
    };

    Config config_;

    Eigen::VectorX<T> compute_direction(const Eigen::VectorX<T>& g, const std::deque<StepHistory>& history);

    T line_search(ObjectiveFunc<T> func, 
                  const Eigen::VectorX<T>& x, 
                  const Eigen::VectorX<T>& g, 
                  const Eigen::VectorX<T>& d,
                  T initial_step);
    
    Eigen::VectorX<T> estimate_gradient(ObjectiveFunc<T> func, const Eigen::VectorX<T>& x);
};

} // namespace openrisk::core
