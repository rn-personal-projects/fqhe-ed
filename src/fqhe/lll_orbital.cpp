#include "fqhe/lll_orbital.hpp"
#include <cmath>
#include <stdexcept>
#include <numbers>

namespace fqhe {

LLLOrbital::LLLOrbital(std::size_t m, std::size_t grid_size)
    : m_(m), grid_size_(grid_size), r_max_(std::sqrt(2.0 * m) + 5.0) {
    d_r_ = r_max_ / grid_size_;
    d_theta_ = 2 * std::numbers::pi / grid_size_;
    norm_const = 1.0 / std::sqrt(2 * std::numbers::pi * std::pow(2.0, m) * std::tgamma(m + 1));
    this->computed_ = false;
}

LLLOrbital::LLLOrbital(std::size_t m, std::size_t grid_size, double r_max)
    : m_(m), grid_size_(grid_size), r_max_(r_max) {
    d_r_ = r_max_ / grid_size_;
    d_theta_ = 2 * std::numbers::pi / grid_size_;
    norm_const = 1.0 / std::sqrt(2 * std::numbers::pi * std::pow(2.0, m) * std::tgamma(m + 1));
    this->computed_ = false;
}

Geometry LLLOrbital::geometry() const {
    return Geometry::DISK;
}

void LLLOrbital::compute() {
    psi.resize(grid_size_, grid_size_);
    for(std::size_t i = 0; i < grid_size_; ++i) {
        for (std::size_t j = 0; j < grid_size_; ++j) {
            double r = i * d_r_;
            double theta = j * d_theta_;
            psi(i, j) = evaluate(r, theta);
        }
    }
    this->computed_ = true;
}

bool LLLOrbital::is_computed() const {
    return computed_;
}

std::optional<LLLOrbital::Complex> LLLOrbital::overlap(const LLLOrbital& other) const {
    if (!this->is_computed() || !other.is_computed()) {
        return std::nullopt;
    }
    if(!(this->grid_size_ == other.grid_size_) || !(this->r_max_ == other.r_max_)) {
        throw std::runtime_error("Grid mismatch.");
    }
    return (this->psi.array() * other.psi.array().conjugate()).sum() * d_r_ * d_theta_;
}

LLLOrbital::RealVector LLLOrbital::density_profile(const ComplexVector& positions) const {
    RealVector density(positions.size());
    for(int i = 0; i < positions.size(); ++i) {
        Complex val = evaluate(positions[i]);
        density[i] = std::norm(val);
    }
    return density;
}

LLLOrbital::Complex LLLOrbital::evaluate(Complex z) const {
    if (computed_) {
        double r = std::abs(z);
        double theta = std::arg(z);
        if (r > r_max_) return 0;
        std::size_t r_idx = std::min(static_cast<std::size_t>(r / d_r_), grid_size_ - 1);
        std::size_t t_idx = std::min(static_cast<std::size_t>(fmod(fmod(theta, 2 * M_PI) + 2 * M_PI, 2 * M_PI) / d_theta_), grid_size_ - 1);
        return psi(r_idx, t_idx);
    }
    double r = std::abs(z);
    double theta = std::arg(z);
    return evaluate(r, theta);
}

LLLOrbital::Complex LLLOrbital::evaluate(double r, double theta) const {
    if(computed_) {
        if (r > r_max_) return 0;
        std::size_t r_idx = std::min(static_cast<std::size_t>(r / d_r_), grid_size_ - 1);
        std::size_t t_idx = std::min(static_cast<std::size_t>(fmod(fmod(theta, 2 * M_PI) + 2 * M_PI, 2 * M_PI) / d_theta_), grid_size_ - 1);
        return psi(r_idx, t_idx);
    }
    return norm_const * std::pow(Complex(r * cos(theta), r * sin(theta)), (int)m_) * exp(-0.5 * r * r);
}

std::size_t LLLOrbital::angular_momentum() const noexcept {
    return m_;
}

LLLOrbital::Complex LLLOrbital::overlap(const Wavefunction& other) const {
    const auto* other_lll = dynamic_cast<const LLLOrbital*>(&other);
    if(other_lll) {
        auto result = overlap(*other_lll);
        if (result) {
            return *result;
        }
    }

    return {0.0, 0.0};
}

}