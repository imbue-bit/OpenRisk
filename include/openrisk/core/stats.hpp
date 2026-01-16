#pragma once
#include "concepts.hpp"
#include <vector>
#include <numeric>
#include <cmath>
#include <algorithm>

namespace openrisk::core {

template <FloatingPoint T = double>
class Statistics {
public:
    static T mean(const Eigen::VectorX<T>& data) {
        return data.mean();
    }

    static T variance(const Eigen::VectorX<T>& data, bool unbiased = true) {
        if (data.size() < 2) return 0.0;
        T avg = mean(data);
        T sum_sq = (data.array() - avg).square().sum();
        return sum_sq / (data.size() - (unbiased ? 1 : 0));
    }

    static T standard_deviation(const Eigen::VectorX<T>& data) {
        return std::sqrt(variance(data));
    }

    static T skewness(const Eigen::VectorX<T>& data) {
        T n = static_cast<T>(data.size());
        if (n < 3) return 0.0;
        T avg = mean(data);
        T std_dev = standard_deviation(data);
        T m3 = (data.array() - avg).cube().sum() / n;
        return m3 / std::pow(std_dev, 3);
    }

    static T kurtosis(const Eigen::VectorX<T>& data, bool excess = true) {
        T n = static_cast<T>(data.size());
        if (n < 4) return 0.0;
        T avg = mean(data);
        T std_dev = standard_deviation(data);
        T m4 = (data.array() - avg).pow(4).sum() / n;
        T k = m4 / std::pow(std_dev, 4);
        return excess ? k - 3.0 : k;
    }
};

} // namespace openrisk::core
