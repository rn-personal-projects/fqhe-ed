#include "fqhe/orbital_interaction.hpp"

namespace fqhe {

using Complex = std::complex<double>;

OrbitalInteraction::OrbitalInteraction(
        std::shared_ptr<DiskLLLWavefunctions> wfs, double alpha, double lambda)
        : wfs_(wfs), alpha_(alpha), lambda_(lambda), cache_({}) {
    m_max_ = wfs_->max_angular_momentum();
    wfs_computed_ = wfs_->is_computed();
    elements_computed_ = false;
};

Complex OrbitalInteraction::evaluate_interaction(Complex z1, Complex z2) const {
    if(z1 == z2) throw std::runtime_error("Points in same location.");
    double distance = std::abs(z1 - z2);
    if (distance < 1e-10) distance = 1e-10;
    return lambda_ * std::pow(distance, -alpha_);
}

Complex OrbitalInteraction::evaluate_interaction(
        double r1, double r2, double theta1, double theta2) const {
    if(r1 == r2 && theta1 == theta2) throw std::runtime_error("Points in same location.");
    double distance = std::sqrt(r1 * r1 + r2 * r2 - 2 * r1 * r2 * std::cos(theta1 - theta2));
    if (distance < 1e-10) distance = 1e-10;
    return lambda_ * std::pow(distance, -alpha_);
}

bool OrbitalInteraction::is_cached(std::size_t m, std::size_t n, std::size_t p, std::size_t q) const {
    return cache_.find({m, n, p, q}) != cache_.end();
}


void OrbitalInteraction::compute_all_wfs(std::optional<std::size_t> grid_size) {
    if(wfs_->is_computed()) return;
    if(wfs_->can_compute()) wfs_->compute();
    else if(!grid_size) throw std::runtime_error("Attempt to compute wavefunctions without grid size.");
    wfs_->enable_compute(*grid_size);
    wfs_->compute();
};

void OrbitalInteraction::clear_cache() {
    cache_.clear();
}

std::tuple<double, double> OrbitalInteraction::parameters() const {
    return {lambda_, alpha_};
}

Complex OrbitalInteraction::compute_element(std::size_t m, std::size_t n, std::size_t p, std::size_t q) const {
    using Real = double;
    using Key = std::tuple<std::size_t, std::size_t, std::size_t, std::size_t>;

    if (m + n != p + q) {
        Key key = {m, n, p, q};
        cache_[key] = 0.0;
        return 0.0;
    }

    const std::array<std::pair<Key, Complex (*)(Complex)>, 8> permutations = {{
          {{m, n, p, q}, [](Complex z) { return z; }},
          {{n, m, p, q}, [](Complex z) { return -z; }},
          {{m, n, q, p}, [](Complex z) { return -z; }},
          {{n, m, q, p}, [](Complex z) { return z; }},
          {{p, q, m, n}, [](Complex z) { return std::conj(z); }},
          {{q, p, m, n}, [](Complex z) { return -std::conj(z); }},
          {{p, q, n, m}, [](Complex z) { return -std::conj(z); }},
          {{q, p, n, m}, [](Complex z) { return std::conj(z); }},
    }};
    for (const auto& [key, transform] : permutations) {
        auto it = cache_.find(key);
        if (it != cache_.end()) {
            Complex result = transform(it->second);
            cache_[{m, n, p, q}] = result;
            return result;
        }
    }

    auto get_m = wfs_->get_orbital(m);
    auto get_n = wfs_->get_orbital(n);
    auto get_p = wfs_->get_orbital(p);
    auto get_q = wfs_->get_orbital(q);
    if(!get_m || !get_n || !get_p || !get_q) {
        throw std::runtime_error("Error accessing orbital.");
    }
    auto m_orb = *get_m;
    auto n_orb = *get_n;
    auto p_orb = *get_p;
    auto q_orb = *get_q;
    if (!m_orb->is_computed() || !n_orb->is_computed() || !p_orb->is_computed() || !q_orb->is_computed()) {
        throw std::runtime_error("Some orbitals not precomputed.");
    }
    if (m_orb->grid_size() != n_orb->grid_size() || m_orb->grid_size() != p_orb->grid_size() || m_orb->grid_size() != q_orb->grid_size()) {
        throw std::runtime_error("Grid size mismatch.");
    }
    if (m_orb->r_max() != n_orb->r_max() || m_orb->r_max() != p_orb->r_max() || m_orb->r_max() != q_orb->r_max()) {
        throw std::runtime_error("r_max mismatch.");
    }
    std::size_t N = m_orb->grid_size();
    Real d_r = m_orb->d_r();
    Real d_theta = m_orb->d_theta();
    Eigen::VectorXd r_vals(N), theta_vals(N);
    for (std::size_t i = 0; i < N; ++i) {
        r_vals(i) = (i + 0.5) * d_r;
        theta_vals(i) = i * d_theta;
    }
    Complex total = 0.0;
    for (std::size_t i1 = 0; i1 < N; ++i1) {
        double r1 = r_vals(i1);
        for (std::size_t j1 = 0; j1 < N; ++j1) {
            double theta1 = theta_vals(j1);
            Complex z1 = std::polar(r1, theta1);
            Complex psi_m_conj = std::conj(m_orb->evaluate(r1, theta1));
            Complex psi_p_val  = p_orb->evaluate(r1, theta1);
            for (std::size_t i2 = 0; i2 < N; ++i2) {
                double r2 = r_vals(i2);
                for (std::size_t j2 = 0; j2 < N; ++j2) {
                    double theta2 = theta_vals(j2);
                    Complex z2 = std::polar(r2, theta2);
                    if (i1 == i2 && j1 == j2) continue;
                    Complex psi_n_conj = std::conj(n_orb->evaluate(r2, theta2));
                    Complex psi_q_val  = q_orb->evaluate(r2, theta2);
                    double jacobian = r1 * r2 * d_r * d_r * d_theta * d_theta;
                    Complex interaction = evaluate_interaction(z1, z2);
                    Complex term = psi_m_conj * psi_n_conj * interaction * psi_p_val * psi_q_val * jacobian;

                    if (std::isnan(std::real(term)) || std::isnan(std::imag(term)) || 
                        std::isnan(std::real(interaction)) || std::isnan(std::imag(interaction))) {
                        continue;
                    }
                    
                    total += term;
                }
            }
        }
    }

    return total;
}

Complex OrbitalInteraction::operator()(std::size_t m, std::size_t n, std::size_t p, std::size_t q) const {
    Complex res {};
    if(!is_cached(m, n, p, q)) { res = compute_element(m, n, p, q); }
    else res = cache_[{m, n, p, q}];
    return res;
}

std::shared_ptr<DiskLLLWavefunctions> OrbitalInteraction::wavefunctions() const {
    return wfs_;
}

void OrbitalInteraction::compute_all_elements() {
    for (std::size_t m = 0; m <= m_max_; ++m) {
        for (std::size_t n = 0; n <= m_max_; ++n) {
            for (std::size_t p = 0; p <= m_max_; ++p) {
                for (std::size_t q = 0; q <= m_max_; ++q) {
                    if (!is_cached(m, n, p, q)) {
                        compute_element(m, n, p, q);
                    }
                }
            }
        }
    }
    elements_computed_ = true;
}

}