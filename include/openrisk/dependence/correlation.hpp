#pragma once
#include "../core/concepts.hpp"
#include <algorithm>
#include <vector>

namespace openrisk::dependence {

template <core::FloatingPoint T = double>
class Correlation {
public:
    /**
     * @brief Pearson 线性相关系数
     */
    static T pearson(const Eigen::VectorX<T>& x, const Eigen::VectorX<T>& y) {
        T x_mean = x.mean();
        T y_mean = y.mean();
        auto x_centered = x.array() - x_mean;
        auto y_centered = y.array() - y_mean;
        
        T numerator = (x_centered * y_centered).sum();
        T denominator = std::sqrt(x_centered.square().sum() * y_centered.square().sum());
        
        return (denominator > 0) ? numerator / denominator : 0.0;
    }

    /**
     * @brief Spearman 秩相关系数
     */
    static T spearman(const Eigen::VectorX<T>& x, const Eigen::VectorX<T>& y) {
        return pearson(rank_transform(x), rank_transform(y));
    }

    /**
     * @brief Kendall's Tau
     */
    static T kendall_tau(const Eigen::VectorX<T>& x, const Eigen::VectorX<T>& y) {
        std::size_t n = x.size();
        if (n < 2) return 0.0;

        long long concordant = 0;
        long long discordant = 0;

        for (std::size_t i = 0; i < n; ++i) {
            for (std::size_t j = i + 1; j < n; ++j) {
                T x_diff = x(i) - x(j);
                T y_diff = y(i) - y(j);
                T prod = x_diff * y_diff;
                
                if (prod > 0) concordant++;
                else if (prod < 0) discordant++;
            }
        }
        return static_cast<T>(concordant - discordant) / (0.5 * n * (n - 1));
    }

private:
    static Eigen::VectorX<T> rank_transform(const Eigen::VectorX<T>& vec) {
        std::size_t n = vec.size();
        std::vector<std::pair<T, std::size_t>> data(n);
        for (std::size_t i = 0; i < n; ++i) data[i] = {vec(i), i};
        
        std::sort(data.begin(), data.end());
        
        Eigen::VectorX<T> ranks(n);
        for (std::size_t i = 0; i < n; ++i) {
            ranks(data[i].second) = static_cast<T>(i + 1);
        }
        return ranks;
    }
};

} // namespace openrisk::dependence
