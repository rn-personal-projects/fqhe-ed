#pragma once

#include <cstdint>
#include <vector>
#include <bitset>

namespace fqhe {

/**
 * Bitstring representation of Fock basis states.
 */
class BasisState {
public:
    /**
     * @brief Constructor with explicit bit representation.
     * @param bits Occupation number bitstring; each index i corresponds to the ith angular
     * momentum state being occupied.
     */
    explicit BasisState(uint64_t bits);

    /**
     * @brief Check if orbital m is occupied
     * @param m Orbital index
     * @return True if m is occupied, false otherwise
     */
    bool occupied(int m) const;

    /**
     * @brief Getter of number of particles in this state
     * @return Number of particles
     */
    int count_particles() const;

    /**
     * @brief Getter of total angular momentum
     * @return Total angular momentum: sum over occupied states of their angular momentum
     */
    int total_angular_momentum() const;

    /**
     * @brief Generate all basis states with N particles and M orbitals
     * @param N Number of particles
     * @param M Number of orbitals
     * @return Vector of basis states
     */
    static std::vector<BasisState> generate(int N, int M);

    /**
     * @brief Generate basis states with N particles, M orbitals, total
     * angular momentum Lz
     * @param N Number of particles
     * @param M Number of orbitals
     * @param Lz Total angular momentum
     * @return Vector of basis states with fixed Lz
     */
    static std::vector<BasisState> generate_fixed_Lz(int N, int M, int Lz);

    /**
     * @brief Check if this state differs from another by exactly two orbitals
     * @param other Other basis state
     * @param i Annihilated orbital index in this state
     * @param j Second annihilated orbital index in this state
     * @param k Created orbital index in other state
     * @param l Created orbital index in other state
     * @return True if states differ by exactly two orbitals
     */
    bool differs_by_two(const BasisState& other, int& i, int& j, int& k, int& l) const;

    /**
     * @brief Getter of raw bit representation
     * @return Bitstring
     */
    uint64_t raw() const noexcept;

    /**
     * @brief Equality comparison
     */
    bool operator==(const BasisState& other) const;

    /**
     * @brief Inequality comparison
     */
    bool operator!=(const BasisState& other) const;

    /**
     * @brief Less than comparison.
     */
    bool operator<(const BasisState& other) const;

private:
    uint64_t bits_;
};
}