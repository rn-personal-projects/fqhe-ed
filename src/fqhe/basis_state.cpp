#include "fqhe/basis_state.hpp"

namespace fqhe {

BasisState::BasisState(uint64_t bits) : bits_(bits) {}

bool BasisState::occupied(int m) const {
    return (bits_ >> 1) & m;
}

int BasisState::count_particles() const {
    return std::popcount(bits_);
}

int BasisState::total_angular_momentum() const {
    int sum = 0;
    uint64_t bits = bits_;
    for (int i = 0; bits != 0; ++i) {
        if (bits & 1)
            sum += i;
        bits >>= 1;
    }
    return sum;
}

namespace {

void backtrack(
       int n_particles,
       int m_max,
       int pos,
       uint64_t state,
       std::vector<BasisState> &states
       ) {
    if (n_particles == 0) {
        states.emplace_back(state);
        return;
    }
    if (pos >= m_max) return;
    backtrack(n_particles-1, m_max, pos+1, state | (1ULL << pos), states);
    backtrack(n_particles, m_max, pos+1, state, states);
}

void backtrack_fixedLz(
        int n_particles,
        int m_max,
        int pos,
        uint64_t state,
        int sum,
        int target,
        std::vector<BasisState> &states
        ) {
    if (n_particles == 0) {
        if (sum == target) states.emplace_back(state);
        return;
    }
    if (pos >= m_max) return;
    if (sum + pos <= target) {
        backtrack_fixedLz(n_particles-1, m_max, pos+1, state | (1ULL << pos), sum + pos, target, states);
    }
    backtrack_fixedLz(n_particles, m_max, pos+1, state, sum, target, states);
}

}
std::vector<BasisState> BasisState::generate(int N, int M) {
    std::vector<BasisState> states;
    backtrack(N, M, 0, 0, states);
    return states;
}

std::vector<BasisState> BasisState::generate_fixed_Lz(int N, int M, int Lz) {
    std::vector<BasisState> states;
    backtrack_fixedLz(N, M, 0, 0, 0, Lz,  states);
    return states;
}

bool BasisState::differs_by_two(const BasisState& other, int& i, int&
j, int& k, int& l) const {
    uint64_t b1 = bits_;
    uint64_t b2 = other.bits_;
    uint64_t diff = b1 ^ b2;
    if (std::popcount(diff) != 2 * 2)
        return false;

    uint64_t created = b2 & ~b1;
    uint64_t annihilated = b1 & ~b2;

    if (std::popcount(created) != 2 || std::popcount(annihilated) != 2)
        return false;


    i = std::countr_zero(created);
    created &= ~(1ULL << i);
    j = std::countr_zero(created);


    k = std::countr_zero(annihilated);
    annihilated &= ~(1ULL << k);
    l = std::countr_zero(annihilated);

    return true;
}

uint64_t BasisState::raw() const noexcept {
    return bits_;
}

bool BasisState::operator==(const BasisState& other) const {
    return bits_ == other.bits_;
}

bool BasisState::operator!=(const BasisState& other) const {
    return bits_ != other.bits_;
}

bool BasisState::operator<(const BasisState& other) const {
    return bits_ < other.bits_;
}

}