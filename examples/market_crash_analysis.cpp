#include "openrisk/crash/lppl.hpp"
#include "openrisk/time_series/garch.hpp"
#include "openrisk/tail/var.hpp"
#include <iostream>
#include <iomanip>

int main() {
    using namespace openrisk;

    const int n_days = 100;
    Eigen::VectorXd t(n_days);
    Eigen::VectorXd log_prices(n_days);
    
    for(int i = 0; i < n_days; ++i) {
        t(i) = i;
        log_prices(i) = 4.0 + 0.02 * i + 0.05 * std::cos(0.1 * i);
    }

    std::cout << "OpenRisk Market Analysis Report:" << std::endl;

    auto lppl_res = crash::LPPLCalibrator<double>::calibrate(t, log_prices, 99.0);
    std::cout << "[LPPL] Predicted Crash Time (tc): " << std::fixed << std::setprecision(2) << lppl_res.tc << std::endl;
    std::cout << "[LPPL] Bubble Strength (m): " << lppl_res.m << std::endl;

    Eigen::VectorXd returns = log_prices.bottomRows(n_days-1) - log_prices.topRows(n_days-1);
    auto garch_params = time_series::calibrate_garch(returns);
    std::cout << "[GARCH] Omega: " << garch_params.omega << " Alpha: " << garch_params.alpha << " Beta: " << garch_params.beta << std::endl;

    double var_95 = tail::RiskMetrics<double>::historical_var(returns, 0.95);
    std::cout << "[Tail] 95% Historical VaR: " << var_95 << std::endl;

    return 0;
}
