#include "fqhe/disk_lll_wavefunctions.hpp"
#include "fqhe/lll_orbital.hpp"
#include <cmath>
#include <stdexcept>

namespace fqhe {

DiskLLLWavefunctions::DiskLLLWavefunctions(std::size_t max_m)
    : max_m_(max_m), grid_size_(0), ready_to_compute_(false), computed_(false), r_max_(std::sqrt(2.0 * max_m) + 5.0) {
    orbitals_.reserve(max_m + 1);
}

DiskLLLWavefunctions::DiskLLLWavefunctions(std::size_t max_m, double r_max)
    : max_m_(max_m), grid_size_(0), ready_to_compute_(false), computed_(false), r_max_(r_max) {
    orbitals_.reserve(max_m + 1);
}

DiskLLLWavefunctions::DiskLLLWavefunctions(std::size_t max_m, double r_max, std::size_t grid_size)
    : max_m_(max_m), grid_size_(grid_size), ready_to_compute_(true), computed_(false), r_max_(r_max) {
    orbitals_.reserve(max_m + 1);
}

std::size_t DiskLLLWavefunctions::max_angular_momentum() const noexcept {
    return max_m_;
}

std::size_t DiskLLLWavefunctions::num_orbitals() const noexcept {
    return max_m_ + 1;
}

std::optional<std::shared_ptr<LLLOrbital>> DiskLLLWavefunctions::get_orbital(std::size_t m) const {
    if (m <= max_m_ && computed_) {
        return orbitals_[m];
    }
    return std::nullopt;
}

const std::vector<std::shared_ptr<LLLOrbital>>& DiskLLLWavefunctions::get_all_orbitals() const {
    return orbitals_;
}

void DiskLLLWavefunctions::enable_compute(std::size_t grid_size) {
    if (!computed_) {
        grid_size_ = grid_size;
        ready_to_compute_ = true;
    }
}

void DiskLLLWavefunctions::compute() {
    if (!ready_to_compute_) {
        throw std::runtime_error("Grid size not yet specified for LLL orbital computation.");
    }
    if (computed_) {
        return;
    }

    orbitals_.clear();
    for (std::size_t m = 0; m <= max_m_; ++m) {
        auto orbital = std::make_shared<LLLOrbital>(m, grid_size_, r_max_);
        orbital->compute();
        orbitals_.push_back(orbital);
    }
    computed_ = true;
}

bool DiskLLLWavefunctions::can_compute() const {
    return ready_to_compute_ && !computed_;
}

bool DiskLLLWavefunctions::is_computed() const {
    return computed_;
}

DiskLLLWavefunctions::Complex DiskLLLWavefunctions::evaluate(std::size_t m, Complex z) const {
    if (!computed_) {
        return {0.0, 0.0};
    }
    auto orbital_opt = get_orbital(m);
    if (orbital_opt) {
        return (*orbital_opt)->evaluate(z);
    }
    return {0.0, 0.0};
}

Geometry DiskLLLWavefunctions::geometry() const {
    return Geometry::DISK;
}

}