#include "openrisk/crash/lppl.hpp"
#include <iostream>
#include <vector>
#include <Eigen/Dense>
#include <cmath>

int main() {
    std::cout << "================================================" << std::endl;
    std::cout << "   OpenRisk Market Analysis Report Demo         " << std::endl;
    std::cout << "================================================" << std::endl;

    const int n = 120;
    Eigen::VectorXd t(n);
    Eigen::VectorXd log_p(n);
    
    for (int i = 0; i < n; ++i) {
        t(i) = static_cast<double>(i);
        double noise = 0.02 * (std::rand() / (double)RAND_MAX);
        log_p(i) = 4.0 + 0.01 * std::pow(130.0 - i, 0.5) + noise; 
    }

    try {
        openrisk::crash::LPPLCalibrator<double> calibrator;
        double t_last = t(n-1);
        
        std::cout << "[INFO] Running LPPL Calibration..." << std::endl;
        auto params = calibrator.calibrate(t, log_p, t_last);

        std::cout << "------------------------------------------------" << std::endl;
        std::cout << "[LPPL] Predicted Crash Time (tc): " << params.tc << std::endl;
        std::cout << "[LPPL] Bubble Strength (m):       " << params.m << std::endl;
        std::cout << "[LPPL] Power Law Magnitude (B):   " << params.B << std::endl;
        std::cout << "[LPPL] Oscillation Amp (C):       " << params.C << std::endl;
        std::cout << "------------------------------------------------" << std::endl;

        if (params.tc > t_last && params.tc < t_last + 30) {
            std::cout << "  WARNING: Critical market regime detected near t=" << params.tc << std::endl;
        } else {
            std::cout << "INFO: No immediate crash regime detected." << std::endl;
        }

    } catch (const std::exception& e) {
        std::cerr << "Error during LPPL analysis: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
