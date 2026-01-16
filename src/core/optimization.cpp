#include "openrisk/core/lbfgs.hpp"
#include <iostream>

namespace openrisk::core {

template <FloatingPoint T>
Eigen::VectorX<T> LBFGSOptimizer<T>::estimate_gradient(ObjectiveFunc<T> func, const Eigen::VectorX<T>& x) {
    Eigen::VectorX<T> g = Eigen::VectorX<T>::Zero(x.size());
    const T h = 1e-7;
    for (int i = 0; i < x.size(); ++i) {
        Eigen::VectorX<T> x_plus = x;
        Eigen::VectorX<T> x_minus = x;
        x_plus(i) += h;
        x_minus(i) -= h;
        g(i) = (func(x_plus) - func(x_minus)) / (2.0 * h);
    }
    return g;
}

template <FloatingPoint T>
Eigen::VectorX<T> LBFGSOptimizer<T>::compute_direction(const Eigen::VectorX<T>& g, const std::deque<StepHistory>& history) {
    Eigen::VectorX<T> q = g;
    std::vector<T> alphas(history.size());

    for (int i = static_cast<int>(history.size()) - 1; i >= 0; --i) {
        alphas[i] = history[i].rho * history[i].s.dot(q);
        q -= alphas[i] * history[i].y;
    }

    if (!history.empty()) {
        const auto& last = history.back();
        T gamma = last.s.dot(last.y) / last.y.dot(last.y);
        q *= gamma;
    }

    for (int i = 0; i < static_cast<int>(history.size()); ++i) {
        T beta = history[i].rho * history[i].y.dot(q);
        q += history[i].s * (alphas[i] - beta);
    }

    return -q;
}

template <FloatingPoint T>
T LBFGSOptimizer<T>::line_search(ObjectiveFunc<T> func, const Eigen::VectorX<T>& x, const Eigen::VectorX<T>& g, const Eigen::VectorX<T>& d, T step) {
    const T c1 = config_.wolfe_c1;
    const T phi_0 = func(x);
    const T phi_prime_0 = g.dot(d);
    
    for (int i = 0; i < 20; ++i) {
        if (func(x + step * d) <= phi_0 + c1 * step * phi_prime_0) {
            return step;
        }
        step *= 0.5;
    }
    return step;
}

template <FloatingPoint T>
OptimizationResult<T> LBFGSOptimizer<T>::minimize(
    ObjectiveFunc<T> func, const Eigen::VectorX<T>& x0,
    const Eigen::VectorX<T>& /*lb*/, const Eigen::VectorX<T>& /*ub*/) {
    
    OptimizationResult<T> res;
    res.x_best = x0;
    res.converged = false;
    
    std::deque<StepHistory> history;
    Eigen::VectorX<T> x = x0;
    Eigen::VectorX<T> g = estimate_gradient(func, x);

    for (int iter = 0; iter < config_.max_iter; ++iter) {
        if (g.norm() < config_.g_tol) {
            res.converged = true;
            break;
        }

        Eigen::VectorX<T> d = compute_direction(g, history);
        T alpha = line_search(func, x, g, d, config_.step_alpha);
        
        Eigen::VectorX<T> x_next = x + alpha * d;
        Eigen::VectorX<T> g_next = estimate_gradient(func, x_next);

        Eigen::VectorX<T> s = x_next - x;
        Eigen::VectorX<T> y = g_next - g;
        T sy = s.dot(y);
        if (sy > 1e-10) { // 保持 Hessian 正定性
            if (history.size() >= static_cast<std::size_t>(config_.m)) history.pop_front();
            history.push_back({s, y, 1.0 / sy});
        }

        x = x_next;
        g = g_next;
        res.iterations = iter;
    }

    res.x_best = x;
    res.min_value = func(x);
    return res;
}

template class LBFGSOptimizer<double>;
template class LBFGSOptimizer<float>;

} // namespace openrisk::core
