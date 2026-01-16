#pragma once
#include <concepts>
#include <Eigen/Dense>

namespace openrisk::core {

/**
 * @brief 约束类型必须为浮点数 (float, double, long double)
 */
template <typename T>
concept FloatingPoint = std::is_floating_point_v<T>;

/**
 * @brief 约束类型必须为 Eigen 的列向量
 */
template <typename T>
concept EigenVector = requires(T v) {
    typename T::Scalar;
    { v.size() } -> std::convertible_to<std::size_t>;
    { v(0) } -> std::convertible_to<typename T::Scalar>;
};

} // namespace openrisk::core
