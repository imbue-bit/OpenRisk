#pragma once
#include "concepts.hpp"
#include <random>
#include <vector>
#include <memory>

namespace openrisk::core {

template <FloatingPoint T = double>
class RandomEngine {
public:
    explicit RandomEngine(uint32_t seed = std::random_device{}()) 
        : engine_(seed) {}

    /**
     * @brief 生成标准正态分布随机向量
     */
    Eigen::VectorX<T> next_normal_vector(std::size_t size) {
        Eigen::VectorX<T> vec(size);
        std::normal_distribution<T> dist(0.0, 1.0);
        for (std::size_t i = 0; i < size; ++i) {
            vec(i) = dist(engine_);
        }
        return vec;
    }

    /**
     * @brief 生成多元正态分布样本向量
     * @param mean 均值向量
     * @param cholesky_l 协方差矩阵的 Cholesky 分解下三角矩阵 L (LL^T = Sigma)
     */
    Eigen::VectorX<T> next_multivariate_normal(const Eigen::VectorX<T>& mean, 
                                              const Eigen::MatrixX<T>& cholesky_l) {
        Eigen::VectorX<T> z = next_normal_vector(mean.size());
        return mean + cholesky_l * z;
    }

    /**
     * @brief 生成 [0, 1] 均匀分布随机数
     */
    T next_uniform() {
        std::uniform_real_distribution<T> dist(0.0, 1.0);
        return dist(engine_);
    }

    void set_seed(uint32_t seed) {
        engine_.seed(seed);
    }

private:
    std::mt19937_64 engine_;
};

} // namespace openrisk::core
