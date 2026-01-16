#pragma once
#include "../core/random.hpp"
#include <cmath>
#include <boost/math/distributions/normal.hpp>
#include <boost/math/distributions/students_t.hpp>

namespace openrisk::dependence {

template <core::FloatingPoint T = double>
class Copula {
public:
    /**
     * @brief 生成 Gaussian Copula 样本
     * @param cholesky_l 相关性矩阵的 Cholesky 分解
     * @param n_samples 样本数量
     */
    static Eigen::MatrixX<T> generate_gaussian_samples(
        const Eigen::MatrixX<T>& cholesky_l, 
        std::size_t n_samples, 
        uint32_t seed = 42) 
    {
        core::RandomEngine<T> rng(seed);
        std::size_t dims = cholesky_l.rows();
        Eigen::MatrixX<T> samples(n_samples, dims);
        
        boost::math::normal_distribution<T> norm_dist(0, 1);

        for (std::size_t i = 0; i < n_samples; ++i) {
            Eigen::VectorX<T> z = cholesky_l * rng.next_normal_vector(dims);
            for (std::size_t j = 0; j < dims; ++j) {
                samples(i, j) = boost::math::cdf(norm_dist, z(j));
            }
        }
        return samples;
    }

    /**
     * @brief 生成 Student-t Copula 样本
     * @param cholesky_l 相关性矩阵的 Cholesky 分解
     * @param df 自由度
     * @param n_samples 样本数量
     */
    static Eigen::MatrixX<T> generate_t_samples(
        const Eigen::MatrixX<T>& cholesky_l, 
        T df, 
        std::size_t n_samples, 
        uint32_t seed = 42) 
    {
        core::RandomEngine<T> rng(seed);
        std::size_t dims = cholesky_l.rows();
        Eigen::MatrixX<T> samples(n_samples, dims);
        
        boost::math::students_t_distribution<T> t_dist(df);
        std::chi_squared_distribution<T> chi_sq(df);
        std::mt19937_64 engine(seed);

        for (std::size_t i = 0; i < n_samples; ++i) {
            Eigen::VectorX<T> z = cholesky_l * rng.next_normal_vector(dims);
            T w = std::sqrt(df / chi_sq(engine));
            for (std::size_t j = 0; j < dims; ++j) {
                samples(i, j) = boost::math::cdf(t_dist, w * z(j));
            }
        }
        return samples;
    }
};

} // namespace openrisk::dependence
