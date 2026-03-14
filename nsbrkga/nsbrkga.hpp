/******************************************************************************
 * nsbrkga.hpp: Non-dominated Sorting Biased Random-Key Genetic Algorithm.
 *
 * This code is released under LICENSE.md.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 ******************************************************************************/

/**
 * \file nsbrkga.hpp
 * \brief Header-only implementation of the Non-dominated Sorting
 *        Biased Random-Key Genetic Algorithm (NS-BRKGA).
 *
 * \details
 * NS-BRKGA is a multi-population, multi-objective metaheuristic that combines
 * biased random-key encoding with non-dominated (Pareto) sorting.  It extends
 * the multi-parent BRKGA framework (MP-BRKGA) to handle multiple simultaneous
 * objectives and includes an Implicit Path Relinking (IPR) post-optimisation
 * step.
 *
 * **Core workflow:**
 * 1. Instantiate `NSBRKGA::NSBRKGA<MyDecoder>` with a decoder, optimization
 *    senses, chromosome size, and an `NsbrkgaParams` object.
 * 2. (Optional) Inject warm-start chromosomes via `setInitialPopulations()`.
 * 3. Call `initialize()` to allocate and decode the initial populations.
 * 4. Loop: call `evolve()`, inspect results via `getIncumbentSolutions()` or
 *    `getChromosome()`, optionally call `exchangeElite()`, `pathRelink()`,
 *    `shake()`, or `reset()`.
 *
 * **Key extension points:**
 * - **Decoder** — a class or functor with
 *   `std::vector<double> decode(Chromosome &, bool rewrite)`.  Must be
 *   thread-safe when `max_threads > 1`.
 * - **Bias function** — controls parent selection probability; choose a
 *   `BiasFunctionType` preset or supply a custom callable via
 *   `setBiasCustomFunction()`.
 * - **Diversity function** — controls elite-set size dynamically; choose a
 *   `DiversityFunctionType` preset or supply a custom callable via
 *   `setDiversityCustomFunction()`.
 * - **Crossover operator** — select `CrossoverType::ROULETTE` (discrete
 *   biased copy) or `CrossoverType::GEOMETRIC` (weighted-average blend).
 * - **Distance function** — used by path relinking to measure chromosome
 *   dissimilarity; subclass `DistanceFunctionBase` or use
 *   `HammingDistance`, `KendallTauDistance`, or `EuclideanDistance`.
 * - **Path relinking type** — choose `PathRelinking::Type::ALLOCATION`,
 *   `PERMUTATION`, or `BINARY_SEARCH`.
 *
 * \see NSBRKGA::Chromosome
 * \see NSBRKGA::NsbrkgaParams
 * \see NSBRKGA::NSBRKGA
 */

#ifndef NSBRKGA_HPP_
#define NSBRKGA_HPP_

// This includes helpers to read and write enums.
#include "third_part/enum_io.hpp"

#include "chromosome.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <deque>
#include <fstream>
#include <functional>
#include <iostream>
#include <limits>
#include <memory>
#include <omp.h>
#include <random>
#include <set>
#include <sstream>
#include <stdexcept>
#include <sys/time.h>
#include <unordered_map>
#include <utility>
#include <vector>

/**
 * \def INLINE
 * \brief Conditionally expands to `inline` when this header is included in
 *        more than one translation unit.
 *
 * \details
 * Because NS-BRKGA is header-only, free functions and explicit template
 * instantiations would violate the One Definition Rule if the same translation
 * unit is compiled into multiple object files.  Define
 * `NSBRKGA_MULTIPLE_INCLUSIONS` before including this header to emit `inline`
 * on those definitions and avoid linker errors.  Note that aggressive inlining
 * can increase object-code size; leave the macro undefined for single-TU
 * builds (the typical case).
 */
#ifdef NSBRKGA_MULTIPLE_INCLUSIONS
#define INLINE inline
#else
#define INLINE
#endif

/**
 * \brief Namespace enclosing the entire NS-BRKGA library.
 *
 * \details
 * All types, free functions, enumerations, and the main algorithm class
 * (`NSBRKGA`) reside in this namespace.  The supporting `PathRelinking`
 * sub-namespace holds enumerations specific to the path relinking procedure.
 *
 * \see NSBRKGA::NSBRKGA       — main algorithm class.
 * \see NSBRKGA::NsbrkgaParams — algorithm hyper-parameters.
 * \see NSBRKGA::Chromosome    — chromosome type definition.
 */
namespace NSBRKGA {

//----------------------------------------------------------------------------//
// Enumerations
//----------------------------------------------------------------------------//

/// Specifies the optimization direction for a single objective.
/// Use `MINIMIZE` for cost-type objectives and `MAXIMIZE` for profit-type
/// objectives.  Pass one `Sense` per objective in the vector given to
/// `NSBRKGA::NSBRKGA()`.
/// \see NSBRKGA::NSBRKGA
enum class Sense {
    MINIMIZE = false, ///< Minimize the objective value.
    MAXIMIZE = true   ///< Maximize the objective value.
};

/// \brief Sub-namespace containing enumerations for the path relinking
///        procedure.
/// \see NSBRKGA::NSBRKGA::pathRelink()
namespace PathRelinking {

/**
 * \brief Specifies the path relinking strategy.
 *
 * \details
 * Each strategy defines how the algorithm moves from a base chromosome toward
 * a guiding chromosome, generating and evaluating a sequence of intermediate
 * solutions along the path.
 *
 * | Value           | Description |
 * |-----------------|-----------------------------------------------------------|
 * | ALLOCATION      | Each allele is directly replaced by the corresponding
 * allele of the guiding chromosome (key-to-key assignment). | | PERMUTATION |
 * The relative rank order induced by the guiding chromosome is imposed on the
 * base chromosome. Suitable when the decoder interprets ranks (permutations). |
 * | BINARY_SEARCH   | Performs a binary search between solutions, halving the
 * remaining distance at each step. |
 *
 */
enum class Type {
    /// Replaces each allele of the base chromosome with the corresponding
    /// allele of the guiding chromosome, one at a time.
    ALLOCATION,

    /// Re-orders alleles in the base chromosome to match the permutation
    /// order induced by the guiding chromosome.  Use when the decoder is
    /// permutation-based.
    PERMUTATION,

    /// Performs a binary search between base and guiding chromosomes,
    /// evaluating the midpoints until the path is sufficiently explored.
    BINARY_SEARCH
};

/**
 * \brief Result status returned by a path relinking call.
 *
 * \details
 * The result codes are ordered by improvement level so that they can be
 * combined with bitwise OR via `operator|=()` to accumulate the best status
 * across multiple relinking runs:
 *
 * | Value            | Meaning                                               |
 * |------------------|-------------------------------------------------------|
 * | NO_IMPROVEMENT   | Relinking finished without improving any solution.    |
 * | ELITE_IMPROVEMENT| An elite-set solution was improved, but not the best. |
 * | BEST_IMPROVEMENT | The overall best incumbent solution was improved.     |
 */
enum class PathRelinkingResult {
    /// Path relinking completed but no improvement was found.
    NO_IMPROVEMENT = 0,

    /// At least one elite solution was improved, but the global best was not.
    ELITE_IMPROVEMENT = 1,

    /// The global best (incumbent) solution was improved.
    BEST_IMPROVEMENT = 3
};

/**
 * \brief Accumulates the highest-ranked `PathRelinkingResult` via bitwise OR.
 *
 * \details
 * Since the enum values are ordered by severity
 * (NO_IMPROVEMENT < ELITE_IMPROVEMENT < BEST_IMPROVEMENT), OR-ing two results
 * always yields the more significant outcome.  Example:
 * \code{.cpp}
 *   result |= PathRelinking::PathRelinkingResult::ELITE_IMPROVEMENT;
 * \endcode
 *
 * \param lhs left-hand operand; updated in place.
 * \param rhs right-hand operand.
 * \return Reference to the updated `lhs`.
 */
inline PathRelinkingResult &operator|=(PathRelinkingResult &lhs,
                                       PathRelinkingResult rhs) {
    lhs = PathRelinkingResult(static_cast<unsigned>(lhs) |
                              static_cast<unsigned>(rhs));
    return lhs;
}
} // namespace PathRelinking

/**
 * \brief Preset bias-function types for parent selection during mating.
 *
 * \details
 * The bias function \f$f(r)\f$ assigns a selection weight to the \f$r\f$-th
 * parent (1-indexed, sorted best-to-worst).  A higher weight increases the
 * probability that parent \f$r\f$ contributes an allele to the offspring.
 * All presets are positive and non-increasing in \f$r\f$, favouring the best
 * parents.
 *
 * | Value      | Formula                                       | Notes |
 * |------------|-----------------------------------------------|----------------------------------|
 * | CONSTANT   | \f$\frac{1}{P}\f$ (\f$P\f$ = total parents)   | Uniform; no
 * preference.          | | CUBIC      | \f$r^{-3}\f$ | Strong elite bias. | |
 * EXPONENTIAL| \f$e^{-r}\f$                                  | Very strong
 * elite bias.          | | LINEAR     | \f$\frac{1}{r}\f$ | Moderate bias. | |
 * LOGINVERSE | \f$\frac{1}{\log(r+1)}\f$                     | Often best in
 * practice.          | | QUADRATIC  | \f$r^{-2}\f$ | Moderate-strong bias. | |
 * SQRT       | \f$r^{-1/2}\f$                                | Biases toward
 * *worse* parents.   | | CBRT       | \f$r^{-1/3}\f$ | Mild bias toward worse
 * parents.  | | CUSTOM     | User-defined via `setBiasCustomFunction()`.   |
 * Must be positive non-increasing. |
 *
 * \see NSBRKGA::NSBRKGA::setBiasCustomFunction()
 */
enum class BiasFunctionType {
    // 1 / num. parents for mating
    /// \f$\frac{1}{\text{num. parents for mating}}\f$
    /// (all individuals have the same probability)
    CONSTANT,

    // r^-3
    /// \f$r^{-3}\f$
    CUBIC,

    // e^-r
    /// \f$e^{-r}\f$
    EXPONENTIAL,

    // 1/r
    /// \f$1/r\f$
    LINEAR,

    // 1 / log(r + 1)
    /// \f$\frac{1}{\log(r + 1)}\f$ (usually works better than other functions)
    LOGINVERSE,

    // r^-2
    /// \f$r^{-2}\f$
    QUADRATIC,

    // r^(-1/2)
    /// \f$r^{-\frac{1}{2}}\f$
    SQRT,

    // r^(-1/3)
    /// \f$r^{-\frac{1}{3}}\f$
    CBRT,

    /// Indicates a custom function supplied by the user.
    CUSTOM
};

/**
 * \brief Preset diversity-function types for adaptive elite-set sizing.
 *
 * \details
 * The diversity function maps a set of elite chromosomes (represented as
 * vectors of doubles) to a scalar diversity score.  After non-dominated
 * sorting, the algorithm adds individuals to the elite set as long as doing
 * so increases the diversity score.
 *
 * | Value                              | Description |
 * |------------------------------------|------------------------------------------------------|
 * | NONE                               | No diversity: elite size =
 * max(num_non_dominated,    | |                                    |
 * min_num_elites), capped at max_num_elites.           | |
 * AVERAGE_DISTANCE_TO_CENTROID       | Average Euclidean distance of each
 * member to the     | |                                    | centroid of the
 * set.                                 | | AVERAGE_DISTANCE_BETWEEN_ALL_PAIRS |
 * Average pairwise Euclidean distance.                 | | POWER_MEAN_BASED |
 * Power-mean aggregation of pairwise distances.        | | CUSTOM |
 * User-defined via `setDiversityCustomFunction()`.     |
 *
 * \see NSBRKGA::NSBRKGA::setDiversityCustomFunction()
 */
enum class DiversityFunctionType {
    NONE,
    AVERAGE_DISTANCE_TO_CENTROID,
    AVERAGE_DISTANCE_BETWEEN_ALL_PAIRS,
    POWER_MEAN_BASED,
    CUSTOM
};

/**
 * \brief Distance-function types used by path relinking to select
 *        sufficiently dissimilar chromosome pairs.
 *
 * \details
 * Before starting path relinking the algorithm measures the distance between
 * candidate chromosome pairs.  Pairs that are too close (below a threshold)
 * are discarded so that relinking explores a non-trivial portion of the
 * search space.
 *
 * | Value       | Description |
 * |-------------|----------------------------------------------------------------|
 * | HAMMING     | Counts allele positions where the discretised bin index | |
 * | differs (see `HammingDistance`).                               | |
 * KENDALL_TAU | Counts pairwise rank inversions between the two chromosomes |
 * |             | (see `KendallTauDistance`).  Suitable for permutation
 * problems.| | EUCLIDEAN   | Standard \f$L_2\f$ distance in \f$[0,1)^n\f$ space
 * | |             | (see `EuclideanDistance`). | | CUSTOM      | User-supplied
 * subclass of `DistanceFunctionBase`.              |
 *
 * \see NSBRKGA::DistanceFunctionBase
 * \see NSBRKGA::HammingDistance
 * \see NSBRKGA::KendallTauDistance
 * \see NSBRKGA::EuclideanDistance
 * \see NSBRKGA::make_distance_function()
 */
enum class DistanceFunctionType {
    HAMMING,     ///< Hamming distance (bin-discretised).
    KENDALL_TAU, ///< Kendall Tau rank-correlation distance.
    EUCLIDEAN,   ///< Euclidean (\f$L_2\f$) distance.
    CUSTOM       ///< Custom user-supplied distance function.
};

/**
 * \brief Crossover operator applied during mating.
 *
 * \details
 * The crossover operator determines how the offspring allele at each gene
 * position is derived from the selected parents.  Both operators are biased
 * by the `BiasFunctionType` applied to the sorted parent ranks, giving
 * higher-ranked (better) parents greater influence.
 *
 * | Value    | Mechanism |
 * |----------|--------------------------------------------------------------------|
 * | ROULETTE | **Discrete.** Each gene independently selects one parent via |
 * |          | roulette-wheel selection weighted by `bias(rank)`, then copies |
 * |          | that parent's allele verbatim.  This is the classical MP-BRKGA |
 * |          | crossover. | | GEOMETRIC| **Continuous.** Each gene computes a
 * weighted average of *all*     | |          | parent alleles.  The weight for
 * parent at rank \f$r\f$ is drawn    | |          | uniformly from
 * \f$[\phi(r),\,\phi(r+1)]\f$ where \f$\phi\f$ is     | |          | the
 * cumulative bias function.  Offspring alleles are in \f$[0,1)\f$| |          |
 * by construction.                                                   |
 *
 * \see NsbrkgaParams::crossover_type
 */
enum class CrossoverType {
    /// Discrete biased roulette crossover: for each gene, a single parent is
    /// chosen via roulette wheel weighted by bias_function(rank) and its allele
    /// is copied.
    ROULETTE,

    /// Biased geometric (weighted-average) crossover: for each gene, every
    /// parent contributes with a random weight drawn from
    /// Uniform(phi(r), phi(r+1)) and the offspring allele is the normalised
    /// weighted average of the parent alleles.
    GEOMETRIC
};

//---------------------------------------------------------------------------//
// Distance functions
//---------------------------------------------------------------------------//

/**
 * \brief Abstract base class for chromosome distance functors.
 *
 * \details
 * Subclass this interface to provide a custom distance metric for path
 * relinking.  The distance is used to identify chromosome pairs that are
 * sufficiently dissimilar to make path relinking worthwhile.
 *
 * **Invariants expected by the library:**
 * - `distance(v, v) == 0` for any vector `v`.
 * - `distance(v1, v2) >= 0` for all `v1`, `v2`.
 * - `v1.size() == v2.size()` at every call site; throw if this is violated.
 *
 * \see NSBRKGA::HammingDistance
 * \see NSBRKGA::KendallTauDistance
 * \see NSBRKGA::EuclideanDistance
 * \see NSBRKGA::make_distance_function()
 */
class DistanceFunctionBase {
  public:
    /// Default constructor.
    DistanceFunctionBase() = default;

    /// Virtual destructor (required for correct polymorphic deletion).
    virtual ~DistanceFunctionBase() = default;

    /**
     * \brief Computes the distance between two chromosome vectors.
     *
     * \param v1 first chromosome (length \f$n\f$, alleles in \f$[0,1)\f$).
     * \param v2 second chromosome (same length as \p v1).
     * \return Non-negative distance scalar; 0 if and only if `v1 == v2`.
     * \throws std::runtime_error if `v1.size() != v2.size()`.
     */
    virtual double distance(const std::vector<double> &v1,
                            const std::vector<double> &v2) const = 0;
};

//---------------------------------------------------------------------------//

/**
 * \brief Hamming distance between two chromosome vectors.
 *
 * \details
 * This functor discretises each allele into a bin index by computing
 * \f$\lfloor x \cdot B \rfloor\f$ where \f$B\f$ is `num_bins`, and then
 * counts the number of positions where the two chromosomes fall into
 * different bins.  The result is an **unnormalised** integer-valued
 * distance in \f$[0, n]\f$ where \f$n\f$ is the chromosome length.
 *
 * **Typical use:** general-purpose chromosomes with continuous alleles
 * that should be treated as approximately categorical after quantisation.
 * The default `num_bins = 2` splits \f$[0,1)\f$ into `{< 0.5, >= 0.5}`.
 *
 * \see NSBRKGA::DistanceFunctionBase
 */
class HammingDistance : public DistanceFunctionBase {
  public:
    /// Number of equal-width bins used to discretise each allele in
    /// \f$[0, 1)\f$.  Larger values increase sensitivity but also noise.
    unsigned num_bins;

    /**
     * \brief Constructs a Hamming functor with the given bin count.
     * \param _num_bins number of discretisation bins; must be \f$\ge 1\f$.
     *        Default: 2 (bisects the unit interval).
     */
    explicit HammingDistance(const double _num_bins = 2)
        : num_bins(_num_bins) {}

    /// Default destructor.
    virtual ~HammingDistance() = default;

    /**
     * \brief Computes the Hamming distance between two chromosome vectors.
     *
     * Counts positions where
     * \f$\lfloor v_1[i] \cdot B \rfloor \ne \lfloor v_2[i] \cdot B \rfloor\f$.
     *
     * \param vector1 first chromosome.
     * \param vector2 second chromosome (must be the same length as \p vector1).
     * \return Unnormalised Hamming distance in \f$[0, n]\f$.
     * \throws std::runtime_error if the vectors have different sizes.
     */
    virtual double distance(const std::vector<double> &vector1,
                            const std::vector<double> &vector2) const {
        if (vector1.size() != vector2.size()) {
            throw std::runtime_error(
                "The size of the vector must be the same!");
        }

        int dist = 0;
        for (std::size_t i = 0; i < vector1.size(); i++) {
            if (unsigned(vector1[i] * this->num_bins) !=
                unsigned(vector2[i] * this->num_bins)) {
                dist++;
            }
        }

        return double(dist);
    }
};

//---------------------------------------------------------------------------//

/**
 * \brief Kendall Tau rank-correlation distance between two chromosomes.
 *
 * \details
 * This functor interprets each chromosome as a ranking of \f$n\f$ items:
 * the item at position \f$i\f$ receives rank according to \f$v[i]\f$.
 * The distance is the number of **concordance–discordance disagreements**
 * (inversions) between the two rank orderings — i.e., the number of pairs
 * \f$(i, j)\f$ with \f$i < j\f$ for which the relative order of item \f$i\f$
 * and item \f$j\f$ differs between the two chromosomes.
 *
 * The result is **unnormalised** and lies in \f$[0, \binom{n}{2}]\f$.
 * Complexity: \f$O(n \log n + n^2)\f$.
 *
 * **Typical use:** permutation-based decoders where allele rank order defines
 * the solution, not the raw allele values.
 *
 * \see NSBRKGA::DistanceFunctionBase
 */
class KendallTauDistance : public DistanceFunctionBase {
  public:
    /// Default constructor.
    KendallTauDistance() {}

    /// Default destructor.
    virtual ~KendallTauDistance() = default;

    /**
     * \brief Computes the Kendall Tau distance between two chromosomes.
     *
     * \param vector1 first chromosome.
     * \param vector2 second chromosome (same length as \p vector1).
     * \return Unnormalised inversion count in \f$[0, \binom{n}{2}]\f$.
     * \throws std::runtime_error if the vectors have different sizes.
     */
    virtual double distance(const std::vector<double> &vector1,
                            const std::vector<double> &vector2) const {
        if (vector1.size() != vector2.size()) {
            throw std::runtime_error(
                "The size of the vector must be the same!");
        }

        const std::size_t size = vector1.size();

        std::vector<std::pair<double, std::size_t>> pairs_v1;
        std::vector<std::pair<double, std::size_t>> pairs_v2;

        pairs_v1.reserve(size);
        std::size_t rank = 0;
        for (std::size_t i = 0; i < vector1.size(); i++) {
            pairs_v1.emplace_back(vector1[i], rank++);
        }

        pairs_v2.reserve(size);
        rank = 0;
        for (std::size_t i = 0; i < vector2.size(); i++) {
            pairs_v2.emplace_back(vector2[i], rank++);
        }

        std::sort(begin(pairs_v1), end(pairs_v1));
        std::sort(begin(pairs_v2), end(pairs_v2));

        unsigned disagreements = 0;
        for (std::size_t i = 0; i + 1 < size; i++) {
            for (std::size_t j = i + 1; j < size; j++) {
                if ((pairs_v1[i].second < pairs_v1[j].second &&
                     pairs_v2[i].second > pairs_v2[j].second) ||
                    (pairs_v1[i].second > pairs_v1[j].second &&
                     pairs_v2[i].second < pairs_v2[j].second)) {
                    disagreements++;
                }
            }
        }

        return double(disagreements);
    }
};

//---------------------------------------------------------------------------//

/**
 * \brief Euclidean (\f$L_2\f$) distance between two chromosome vectors.
 *
 * \details
 * Computes \f$\|v_1 - v_2\|_2 = \sqrt{\sum_i (v_1[i] - v_2[i])^2}\f$.
 * The result is **unnormalised**; for chromosomes with alleles in \f$[0,1)\f$
 * the maximum value is \f$\sqrt{n}\f$ where \f$n\f$ is the chromosome length.
 * Complexity: \f$O(n)\f$.
 *
 * \see NSBRKGA::DistanceFunctionBase
 */
class EuclideanDistance : public DistanceFunctionBase {
  public:
    /// Default constructor.
    EuclideanDistance() {}

    /// Default destructor.
    virtual ~EuclideanDistance() = default;

    /**
     * \brief Computes the Euclidean distance between two chromosomes.
     *
     * \param vector1 first chromosome.
     * \param vector2 second chromosome (same length as \p vector1).
     * \return Euclidean distance \f$\|v_1 - v_2\|_2 \ge 0\f$.
     * \throws std::runtime_error if the vectors have different sizes.
     */
    virtual double distance(const std::vector<double> &vector1,
                            const std::vector<double> &vector2) const {
        if (vector1.size() != vector2.size()) {
            throw std::runtime_error(
                "The size of the vector must be the same!");
        }

        double dist = 0.0;

        for (std::size_t i = 0; i < vector1.size(); i++) {
            dist += (vector1[i] - vector2[i]) * (vector1[i] - vector2[i]);
        }

        return std::sqrt(dist);
    }
};

//---------------------------------------------------------------------------//

/**
 * \brief Factory: creates a concrete `DistanceFunctionBase` from an enum tag.
 *
 * \details
 * Maps a `DistanceFunctionType` value to the corresponding built-in subclass.
 * Pass the returned shared pointer to `pathRelink()` or store it for repeated
 * use.  `CUSTOM` and any unrecognised value fall back to `EuclideanDistance`;
 * to use a truly custom function, construct your subclass directly.
 *
 * \param t the desired distance function type.
 * \return A `std::shared_ptr<DistanceFunctionBase>` owning the new object.
 * \see NSBRKGA::DistanceFunctionType
 */
INLINE std::shared_ptr<DistanceFunctionBase>
make_distance_function(DistanceFunctionType t) {
    switch (t) {
    case DistanceFunctionType::HAMMING:
        return std::make_shared<HammingDistance>();
    case DistanceFunctionType::KENDALL_TAU:
        return std::make_shared<KendallTauDistance>();
    case DistanceFunctionType::EUCLIDEAN:
    default:
        return std::make_shared<EuclideanDistance>();
    }
}

//----------------------------------------------------------------------------//
// Population class.
//----------------------------------------------------------------------------//

/**
 * \brief Internal container for one population of chromosomes.
 *
 * \details
 * `Population` holds a generation's worth of chromosomes together with their
 * decoded fitness vectors.  It provides the sorting, dominance-checking, and
 * elite-set management operations needed by the NSBRKGA engine.
 *
 * \warning This class is **internal to the framework**.  User code should
 * access chromosomes and fitness values through the public NSBRKGA interface
 * (e.g., `NSBRKGA::getChromosome()`, `NSBRKGA::getFitness()`,
 * `NSBRKGA::getIncumbentSolutions()`), not by manipulating Population objects
 * directly.
 *
 * **Multi-objective sorting:**
 * - Fitness entries are sorted by non-dominated rank (front index) and, within
 *   each front, by crowding distance (descending).  The combinedsorting is
 *   performed in \f$O(N \cdot F \cdot \log F)\f$-ish time where N is the
 *   population size and F is the number of fronts.
 * - Non-dominated individuals in the first front (rank 0) are placed at
 *   indices 0..num_non_dominated-1 after sorting.
 *
 * \see NSBRKGA::NSBRKGA
 */
class Population {
  public:
    /** \name Data members */
    //@{
    /// All chromosomes in this population.  Index order matches `fitness`.
    std::vector<Chromosome> population;

    /// Sorted fitness table.  Each entry is `(objective_values, raw_index)`
    /// where `raw_index` identifies the chromosome in `population`.
    /// Entries are sorted best-to-worst after each call to `sortFitness()`.
    std::vector<std::pair<std::vector<double>, unsigned>> fitness;

    /// Minimum number of non-dominated fronts observed across all generations.
    unsigned min_num_fronts;

    /// Maximum number of non-dominated fronts observed across all generations.
    unsigned max_num_fronts;

    /// Number of non-dominated individuals in the current generation
    /// (i.e., the size of Pareto front 0).
    unsigned num_non_dominated;

    /// Total number of distinct non-dominated fronts in the current generation.
    unsigned num_fronts;

    /// Diversity function used to determine the dynamic elite-set size.
    std::function<double(const std::vector<std::vector<double>> &)>
        &diversity_function;

    /// Hard lower bound on the number of elite individuals.
    unsigned min_num_elites;

    /// Hard upper bound on the number of elite individuals.
    unsigned max_num_elites;

    /// Current number of elite individuals (between min and max, inclusive).
    unsigned num_elites;
    //@}

    /** \name Default constructors and destructor */
    //@{
    /**
     * \brief Default constructor.
     *
     * \param chr_size size of chromosome.
     * \param pop_size size of population.
     * \param diversity_function diversity function.
     * \param min_num_elites minimum number of elite individuals.
     * \param max_num_elites maximum number of elite individuals.
     * \throw std::range_error if population size or chromosome size is zero.
     */
    Population(const unsigned chr_size, const unsigned pop_size,
               std::function<double(const std::vector<std::vector<double>> &)>
                   &diversity_function,
               const unsigned min_num_elites, const unsigned max_num_elites)
        : population(pop_size, Chromosome(chr_size, 0.0)), fitness(pop_size),
          min_num_fronts(pop_size), max_num_fronts(1), num_non_dominated(0),
          num_fronts(0), diversity_function(diversity_function),
          min_num_elites(min_num_elites), max_num_elites(max_num_elites),
          num_elites(0) {
        if (pop_size == 0) {
            throw std::range_error("Population size cannot be zero.");
        }

        if (chr_size == 0) {
            throw std::range_error("Chromosome size cannot be zero.");
        }
    }

    /// Copy constructor.
    Population(const Population &other)
        : population(other.population), fitness(other.fitness),
          min_num_fronts(other.min_num_fronts),
          max_num_fronts(other.max_num_fronts),
          num_non_dominated(other.num_non_dominated),
          num_fronts(other.num_non_dominated),
          diversity_function(other.diversity_function),
          min_num_elites(other.min_num_elites),
          max_num_elites(other.max_num_elites), num_elites(other.num_elites) {}

    /// Assignment operator for compliance.
    Population &operator=(const Population &) = default;

    /// Destructor.
    ~Population() = default;
    //@}

    /** \name Simple access methods */
    //@{
    /// Returns the size of each chromosome.
    unsigned getChromosomeSize() const {
        return this->population.front().size();
    }

    /// Returns the size of the population.
    unsigned getPopulationSize() const { return this->population.size(); };

    /**
     * \brief Returns a copy of an allele for a given chromosome.
     * \param chromosome index of desired chromosome.
     * \param allele index of desired allele.
     * \returns a copy of the allele value.
     */
    double operator()(const unsigned chromosome, const unsigned allele) const {
        return this->population[chromosome][allele];
    }

    /**
     * \brief Returns a reference for an allele for a given chromosome.
     *
     * Usually used to set the allele value.
     * \param chromosome index of desired chromosome.
     * \param allele index of desired allele.
     * \returns a reference of the allele value.
     */
    double &operator()(const unsigned chromosome, const unsigned allele) {
        return this->population[chromosome][allele];
    }

    /**
     * \brief Returns a reference to a chromosome.
     * \param chromosome index of desired chromosome.
     * \returns a reference to chromosome.
     */
    Chromosome &operator()(unsigned chromosome) {
        return this->population[chromosome];
    }
    //@}

    /** \name Special access methods
     *
     * These methods REQUIRE fitness to be sorted, and thus a call to
     * `sortFitness()` beforehand.
     */
    //@{
    /**
     * \brief Returns the best fitnesses in this population.
     *
     * This method requires fitness to be sorted, and thus a call to
     * `sortFitness()` beforehand. It performs a non-dominated sort of the
     * fitness, and returns the first (i.e., best) front of solutions'
     * fitnesses.
     *
     * \param senses A reference to a vector of Sense objects that is used for
     *               non-dominated sorting.
     * \return A vector of vector doubles representing the best fitnesses.
     * \throws std::runtime_error if the resulting fronts vector is empty.
     */
    std::vector<std::vector<double>>
    getBestFitnesses(const std::vector<Sense> &senses) const {
        std::vector<std::vector<std::pair<std::vector<double>, unsigned>>>
            fronts =
                Population::nonDominatedSort<unsigned>(this->fitness, senses);

        if (fronts.empty()) {
            throw std::runtime_error("Fronts vector is empty. "
                                     "Have you sorted the fitness?");
        }

        std::vector<std::vector<double>> result(fronts[0].size());
        std::transform(
            fronts[0].begin(), fronts[0].end(), result.begin(),
            [](const std::pair<std::vector<double>, unsigned> &solution) {
                return std::move(solution.first);
            });
        return result;
    }

    /**
     * \brief Returns the best chromosomes in this population.
     *
     * This method requires fitness to be sorted, and thus a call to
     * `sortFitness()` beforehand. It performs a non-dominated sort of the
     * fitness, and then returns the first (i.e., best) front of solutions'
     * chromosomes.
     *
     * \param senses A reference to a vector of Sense objects that is used for
     *               non-dominated sorting.
     * \return A vector of Chromosome representing the best chromosomes.
     * \throws std::runtime_error if the resulting fronts vector is empty.
     */
    std::vector<Chromosome>
    getBestChromosomes(const std::vector<Sense> &senses) const {
        std::vector<std::vector<std::pair<std::vector<double>, unsigned>>>
            fronts =
                Population::nonDominatedSort<unsigned>(this->fitness, senses);

        if (fronts.empty()) {
            throw std::runtime_error("Fronts vector is empty. "
                                     "Have you sorted the fitness?");
        }

        std::vector<Chromosome> result(fronts[0].size());
        std::transform(
            fronts[0].begin(), fronts[0].end(), result.begin(),
            [this](const std::pair<std::vector<double>, unsigned> &item) {
                return this->population[item.second];
            });

        return result;
    }

    /// Returns the fitness of chromosome i.
    std::vector<double> getFitness(const unsigned i) const {
        return this->fitness[i].first;
    }

    /// Returns a reference to the i-th best chromosome.
    Chromosome &getChromosome(unsigned i) {
        return this->population[this->fitness[i].second];
    }

    /// Returns a const reference to the i-th best chromosome.
    const Chromosome &getChromosome(const unsigned i) const {
        return this->population[this->fitness[i].second];
    }

    /**
     * \brief Updates the number of elite individuals
     *        based on diversity function.
     *
     * The method initially sets the number of elite individuals to the
     * minimum value. It calculates the diversity of the first set of elites,
     * then iteratively adds the next individuals and recalculates diversity,
     * updating the number of elites if diversity is increased.
     */
    void updateNumElites() {
        this->num_elites =
            std::max(this->min_num_elites, this->num_non_dominated);
        this->num_elites = std::min(this->num_elites, this->max_num_elites);
        std::vector<std::vector<double>> chromosomes(this->num_elites);
        chromosomes.reserve(this->max_num_elites);

        for (unsigned i = 0; i < this->num_elites; i++) {
            chromosomes[i] = this->getChromosome(i);
        }

        double best_diversity = this->diversity_function(chromosomes);

        for (unsigned i = this->num_elites; i < this->max_num_elites; i++) {
            chromosomes.push_back(this->getChromosome(i));
            double new_diversity = this->diversity_function(chromosomes);

            if (best_diversity < new_diversity) {
                best_diversity = new_diversity;
                this->num_elites = i + 1;
            }
        }
    }

    //@}

    /** \name Other methods */
    //@{
    /**
     * \brief Returns `true` if `a1` is better than `a2`.
     *
     * This method depends on the optimization senses. When the optimization
     * sense is `Sense::MINIMIZE`, `a1 < a2` will return true, otherwise false.
     * The opposite happens for `Sense::MAXIMIZE`. The comparisons are made
     * using an epsilon value to cater for floating point precision issues.
     *
     * \param a1 First comparison value
     * \param a2 Second comparison value
     * \param sense The optimization sense
     * \return true if a1 is better than a2 based on the optimization sense,
     *         otherwise false
     */
    static inline bool betterThan(const double &a1, const double &a2,
                                  const Sense &sense) {
        double epsilon = std::numeric_limits<double>::epsilon();
        return (sense == Sense::MINIMIZE) ? (a1 < a2 - epsilon)
                                          : (a1 > a2 + epsilon);
    }

    /**
     * \brief Checks if `a1` dominates `a2` based on the provided optimization
     * senses.
     *
     * An item `a1` is said to dominate another item `a2` if it
     * is at least as good as `a2` in all respects (as per the
     * senses) and strictly better in at least one respect.
     *
     * \param a1 The first fitness vector.
     * \param a2 The second fitness vector.
     * \param senses The vector of optimization senses
     *               to apply for the domination check.
     * \return true if `a1` dominates `a2` according
     *         to the given optimization senses.
     */
    static inline bool dominates(const std::vector<double> &a1,
                                 const std::vector<double> &a2,
                                 const std::vector<Sense> &senses) {
        bool at_least_as_good = true, better = false;

        for (std::size_t i = 0; i < senses.size() && at_least_as_good; i++) {
            if (Population::betterThan(a2[i], a1[i], senses[i])) {
                // a1 is worse than a2 in at least one objective,
                // thus a1 cannot dominate a2
                at_least_as_good = false;
            } else if (Population::betterThan(a1[i], a2[i], senses[i])) {
                // a1 is better than a2 in at least one objective,
                better = true;
            }
        }

        // a1 is no worse than a2 in all objectives,
        // and is better than a2 in at least one objective,
        // thus a1 dominates a2
        return at_least_as_good && better;
    }

    /**
     * \brief Partitions a set of fitness–payload pairs into non-dominated
     *        fronts (Pareto layers).
     *
     * The method implements a sort-then-insert algorithm that works as
     * follows:
     *
     * 1. **Lexicographic pre-sort.**  All entries are sorted by a
     *    lexicographic comparator that respects the optimisation senses
     *    (MINIMIZE or MAXIMIZE) for each objective.  This guarantees that
     *    when processing entry \e i, every entry that could dominate it has
     *    already been assigned to a front.
     *
     * 2. **Front insertion.**  Each entry (in sorted order) is placed into
     *    the *first* front whose members do not dominate it.  The search
     *    for that front uses a binary search over existing fronts, making
     *    the expected insertion cost O(log F) per element (where F is the
     *    current number of fronts).
     *
     * #### Special-case optimisations
     *
     * | #Objectives | Behaviour |
     * |-------------|-----------|
     * | 0 (empty `senses`) | Returns immediately with an empty result. |
     * | 1 | After sorting, consecutive equal-valued entries share the same
     *       front; a new front is created only when the predecessor
     *       strictly dominates the successor. |
     * | 2 | When checking whether a front dominates the current entry, only
     *       the *last* element in the front is tested (the sorted order
     *       guarantees this is sufficient). |
     * | ≥ 3 | All elements in a front are tested (reverse order) until a
     *         dominator is found or the front is exhausted. |
     *
     * #### Complexity
     *
     * - Time: O(n log n) for the initial sort, plus O(n · F · k) for the
     *   front-insertion phase, where n is the number of entries, F is the
     *   number of fronts produced, and k is the cost per dominance check
     *   (equals `senses.size()` in the worst case).  For 2 objectives,
     *   the per-front check is O(k) (only one element tested), so overall
     *   insertion is O(n · log F · k).
     * - Space: O(n · m) for the temporary sorted copy, where m is the
     *   number of objectives.
     *
     * #### Tie handling
     *
     * Solutions with identical objective vectors are never considered to
     * dominate one another (strict dominance requires being *better* in at
     * least one objective).  They therefore land in the same front.
     *
     * \tparam T  The payload type stored alongside each fitness vector
     *            (e.g. `unsigned` for a population index, or `Chromosome`).
     *
     * \param[in] fitness  Vector of (fitness-vector, payload) pairs to be
     *                     partitioned.  Not modified.
     * \param[in] senses   Optimisation sense for each objective.  Its size
     *                     defines the number of objectives considered.
     *
     * \return A vector of fronts, where front 0 is the non-dominated set.
     *         Each front is a vector of (fitness-vector, payload) pairs in
     *         the order they were inserted (which follows the lexicographic
     *         pre-sort).  Returns an empty vector when `fitness` or
     *         `senses` is empty.
     */
    template <class T>
    static std::vector<std::vector<std::pair<std::vector<double>, T>>>
    nonDominatedSort(
        const std::vector<std::pair<std::vector<double>, T>> &fitness,
        const std::vector<Sense> &senses) {

        std::vector<std::vector<std::pair<std::vector<double>, T>>> result;

        if (fitness.empty() || senses.empty()) {
            return result;
        }

        const std::size_t numObj = senses.size();

        result.reserve(fitness.size());

        // Lexicographic comparator respecting optimisation senses.
        // Captured by const-ref to avoid copying the senses vector.
        auto comp =
            [&senses](const std::pair<std::vector<double>, T> &a,
                      const std::pair<std::vector<double>, T> &b) -> bool {
            for (std::size_t i = 0; i < a.first.size(); i++) {
                if (Population::betterThan(a.first[i], b.first[i], senses[i])) {
                    return true;
                }
                if (Population::betterThan(b.first[i], a.first[i], senses[i])) {
                    return false;
                }
            }

            // a == b
            return false;
        };

        std::vector<std::pair<std::vector<double>, T>> sorted_fitness = fitness;
        std::sort(sorted_fitness.begin(), sorted_fitness.end(), comp);
        result.emplace_back(1, sorted_fitness.front());

        if (numObj == 1) {
            // Single-objective fast path: consecutive solutions either
            // share the current front or start a new one.
            for (std::size_t i = 1; i < sorted_fitness.size(); i++) {
                if (Population::dominates(sorted_fitness[i - 1].first,
                                          sorted_fitness[i].first, senses)) {
                    result.emplace_back(1, sorted_fitness[i]);
                } else {
                    result.back().push_back(sorted_fitness[i]);
                }
            }
        } else { // numObj >= 2
            // Helper: returns true if any solution in result[frontIdx]
            // dominates sorted_fitness[solIdx].  For 2 objectives only
            // the last element of the front needs to be checked (the
            // pre-sort guarantees it is the only possible dominator).
            auto isDominatedByFront = [&result, &sorted_fitness, &senses,
                                       numObj](std::size_t frontIdx,
                                               std::size_t solIdx) -> bool {
                const auto &front = result[frontIdx];

                for (std::size_t j = front.size(); j > 0; j--) {
                    if (Population::dominates(front[j - 1].first,
                                              sorted_fitness[solIdx].first,
                                              senses)) {
                        return true;
                    }

                    // For 2 objectives, only the last element can dominate.
                    if (numObj == 2) {
                        break;
                    }
                }

                return false;
            };

            for (std::size_t i = 1; i < sorted_fitness.size(); i++) {
                // Quick check: if the last front dominates this solution,
                // it must go into a brand-new front.
                if (isDominatedByFront(result.size() - 1, i)) {
                    result.emplace_back(1, std::move(sorted_fitness[i]));
                } else {
                    // Binary search for the first front that does not
                    // dominate the current solution.
                    std::size_t kMin = 0, kMax = result.size();

                    while (kMin < kMax) {
                        std::size_t k = (kMin + kMax) >> 1;

                        if (isDominatedByFront(k, i)) {
                            kMin = k + 1;
                        } else {
                            if (k == kMin) {
                                break;
                            }

                            kMax = k;
                        }
                    }

                    result[kMin].push_back(std::move(sorted_fitness[i]));
                }
            }
        }

        return result;
    }

    /**
     * \brief Sorts a front by descending crowding distance.
     *
     * Implements the NSGA-II crowding-distance assignment and sorts the
     * front so that solutions with larger crowding distance come first.
     *
     * #### Algorithm
     *
     * 1. Initialise a `distance` vector to 0 for every solution, and an
     *    `order` index vector `[0, n)`.
     * 2. For each objective *m*:
     *    - Sort `order` by `fitness[order[i]].first[m]`.
     *    - Boundary solutions (first/last) receive `max()`.
     *    - Interior solutions accumulate the normalised crowding span
     *      `(f[i+1] − f[i−1]) / (fMax − fMin)` unless `max()` was
     *      already assigned.  If the range is degenerate
     *      (`|fMax − fMin| < epsilon`), they also receive `max()`.
     * 3. Build `(distance, index)` pairs and sort descending.
     * 4. Permute `fitness` in-place to match the sorted order.
     *
     * #### Complexity
     *
     * - Time:  O(n · m · log n) — m sorts of n elements.
     * - Space: O(n) — only index and distance vectors; no objective-vector
     *   copies.
     *
     * \tparam T The payload type stored alongside each fitness vector.
     * \param[in,out] fitness  The front to sort.  Modified in place.
     */
    template <class T>
    static void
    crowdingSort(std::vector<std::pair<std::vector<double>, T>> &fitness) {
        const std::size_t n = fitness.size();

        if (n <= 1) {
            return; // Nothing to sort.
        }

        const std::size_t numObj = fitness.front().first.size();

        // distance[i] accumulates the crowding distance for fitness[i].
        std::vector<double> distance(n, 0.0);

        // order[i] is an index into fitness[]; sorted per-objective below.
        std::vector<unsigned> order(n);

        for (unsigned i = 0; i < n; i++) {
            order[i] = i;
        }

        // --- Step 2: accumulate crowding distance for each objective --------
        for (std::size_t m = 0; m < numObj; m++) {
            // Sort indices by the m-th objective value.
            std::sort(order.begin(), order.end(),
                      [&fitness, m](unsigned a, unsigned b) {
                          return fitness[a].first[m] < fitness[b].first[m];
                      });

            const double fMin = fitness[order.front()].first[m];
            const double fMax = fitness[order.back()].first[m];

            // Boundary solutions always receive infinite distance.
            distance[order.front()] = std::numeric_limits<double>::max();
            distance[order.back()] = std::numeric_limits<double>::max();

            // Interior solutions.
            for (std::size_t i = 1; i + 1 < n; i++) {
                const unsigned idx = order[i];

                if (fabs(fMax - fMin) <
                    std::numeric_limits<double>::epsilon()) {
                    // Degenerate range: all solutions are equally spread.
                    distance[idx] = std::numeric_limits<double>::max();
                } else if (distance[idx] < std::numeric_limits<double>::max()) {
                    // Accumulate normalised crowding span from neighbours.
                    distance[idx] += (fitness[order[i + 1]].first[m] -
                                      fitness[order[i - 1]].first[m]) /
                                     (fMax - fMin);
                }
            }
        }

        // --- Step 3: sort by descending (distance, index) -------------------
        // After the last objective loop, `order` holds the sorted-by-last-
        // objective permutation.  Building aux from `order` preserves the
        // same secondary-sort key as the original implementation.
        std::vector<std::pair<double, unsigned>> aux(n);

        for (std::size_t i = 0; i < n; i++) {
            aux[i] = {distance[order[i]], order[i]};
        }

        std::sort(aux.begin(), aux.end(),
                  std::greater<std::pair<double, unsigned>>());

        // --- Step 4: permute fitness in-place -------------------------------
        // Build the target permutation: perm[i] = source index for slot i.
        std::vector<unsigned> perm(n);

        for (std::size_t i = 0; i < n; i++) {
            perm[i] = aux[i].second;
        }

        // Follow cycles to move elements with O(n) swaps and no full copy.
        std::vector<bool> placed(n, false);

        for (std::size_t i = 0; i < n; i++) {
            if (placed[i] || perm[i] == i) {
                continue;
            }

            std::size_t j = i;

            while (!placed[j]) {
                placed[j] = true;

                if (perm[j] != i) {
                    std::swap(fitness[j], fitness[perm[j]]);
                    j = perm[j];
                } else {
                    // Close the cycle: element at perm[j] (== i) is already
                    // in its final position after the preceding swaps.
                    j = perm[j];
                }
            }
        }
    }

    /**
     * \brief Sorts fitness–payload pairs according to the given optimisation
     *        senses.
     *
     * #### Single-objective (one sense)
     *
     * The entries are sorted directly by the single objective value,
     * best-to-worst, using `Population::betterThan`.  The return value
     * is `{n, 1}` where n is the population size (every element forms
     * its own "trivial front" for counting purposes).
     *
     * #### Multi-objective (two or more senses)
     *
     * 1. `nonDominatedSort` partitions the entries into Pareto fronts.
     * 2. Each front is sorted in-place by descending crowding distance
     *    via `crowdingSort`.
     * 3. The sorted fronts are concatenated back into `fitness`
     *    (front 0 first, then front 1, etc.), so the best individuals
     *    occupy the lowest indices.
     *
     * Elements are **moved** (not copied) from the temporary front
     * storage back into `fitness`, avoiding deep copies of the
     * objective vectors.
     *
     * #### Complexity
     *
     * - Single-objective: O(n log n).
     * - Multi-objective: dominated by `nonDominatedSort` and
     *   `crowdingSort`; see their respective documentation.
     *
     * \tparam T  Payload type stored alongside each fitness vector
     *            (e.g. `unsigned` for a population index).
     *
     * \param[in,out] fitness  The fitness–payload pairs to sort.
     *                         Modified in place.
     * \param[in]     senses   Optimisation sense for each objective.
     *
     * \return A pair `{num_fronts, num_non_dominated}` where
     *         `num_fronts` is the total number of fronts produced and
     *         `num_non_dominated` is the size of front 0.  Returns
     *         `{0, 0}` when `fitness` or `senses` is empty.
     */
    template <class T>
    static std::pair<unsigned, unsigned>
    sortFitness(std::vector<std::pair<std::vector<double>, T>> &fitness,
                const std::vector<Sense> &senses) {
        if (fitness.empty() || senses.empty()) {
            return std::make_pair(0, 0);
        }

        // --- Single-objective: direct sort, best-to-worst ----------------
        if (senses.size() == 1) {
            std::sort(fitness.begin(), fitness.end(),
                      [&senses](const std::pair<std::vector<double>, T> &a,
                                const std::pair<std::vector<double>, T> &b) {
                          return Population::betterThan(
                              a.first.front(), b.first.front(), senses.front());
                      });

            return std::make_pair(fitness.size(), 1);
        }

        // --- Multi-objective: non-dominated sort + crowding sort ----------
        auto fronts = Population::nonDominatedSort<T>(fitness, senses);

        // Move each crowding-sorted front back into fitness.
        // std::move (algorithm) avoids deep copies of the fitness vectors;
        // it returns the past-the-end output iterator, which serves as the
        // next write position.
        auto out = fitness.begin();

        for (auto &front : fronts) {
            Population::crowdingSort<T>(front);
            out = std::move(front.begin(), front.end(), out);
        }

        return std::make_pair(fronts.size(), fronts.front().size());
    }

    /**
     * \brief Sorts this population's `fitness` table and updates
     *        population-level statistics.
     *
     * Delegates to the static `sortFitness<unsigned>()` and then
     * refreshes `num_fronts`, `num_non_dominated`, the historical
     * min/max front counters, and the dynamic elite-set size.
     *
     * \param senses Optimisation senses (one per objective).
     */
    void sortFitness(const std::vector<Sense> &senses) {
        auto ret = Population::sortFitness<unsigned>(this->fitness, senses);
        this->num_fronts = ret.first;
        this->num_non_dominated = ret.second;

        this->min_num_fronts = std::min(this->min_num_fronts, this->num_fronts);
        this->max_num_fronts = std::max(this->max_num_fronts, this->num_fronts);

        this->updateNumElites();
    }

    /**
     * \brief Sets the fitness of chromosome.
     * \param chromosome index of chromosome.
     * \param values     fitness values.
     */
    void setFitness(const unsigned chromosome,
                    const std::vector<double> values) {
        this->fitness[chromosome] = std::make_pair(values, chromosome);
    }
    //@}
};

//----------------------------------------------------------------------------//
// NSBRKGA Params class.
//----------------------------------------------------------------------------//

/**
 * \brief Algorithm hyper-parameters for NS-BRKGA.
 *
 * \details
 * All parameters are loaded by `readConfiguration()` or set manually before
 * passing to the `NSBRKGA` constructor.  Invalid combinations are detected
 * and reported via `std::range_error` in the constructor.
 *
 * | Parameter                    | Type     | Valid range          | Guidance |
 * |------------------------------|----------|----------------------|-------------------------------------------------------|
 * | `population_size`            | unsigned | \f$\ge 2\f$          | Typical:
 * 100–500. Larger values slow each generation  | | |          | | but improve
 * coverage.                                 | | `min_elites_percentage`      |
 * double   | \f$(0, 1]\f$         | Fraction of `population_size` giving the
 * minimum      | |                              |          | | elite-set size.
 * Typical: 0.10–0.20.                   | | `max_elites_percentage`      |
 * double   | \f$(0, 1]\f$         | Upper bound for elite-set size fraction.
 * Must be      | |                              |          | | \f$\ge\f$
 * `min_elites_percentage`. Typical: 0.30.     | | `mutation_probability` |
 * double   | \f$[0, 1]\f$         | Per-allele probability of polynomial
 * mutation.        | |                              |          | | 0 disables
 * mutation. Typical: 0.01–0.10.              | | `mutation_distribution`      |
 * double   | \f$> 0\f$            | Polynomial-mutation distribution index
 * \f$\eta_m\f$.  | |                              |          | | Higher =
 * milder perturbation. Typical: 10–100.        | | `num_elite_parents` |
 * unsigned | \f$\ge 1\f$          | Elite parents per offspring. Must be | | |
 * |                      | \f$\le\f$ `total_parents` and \f$\le\f$
 * `min_num_elites`.| | `total_parents`              | unsigned | \f$\ge 2\f$ |
 * Total parents per offspring (elite + non-elite).      | | `bias_type` | enum
 * | `BiasFunctionType`   | Parent selection weighting; `LOGINVERSE`
 * recommended. | | `diversity_type`             | enum     |
 * `DiversityFunctionType`| Elite-set diversity criterion; `NONE` is simplest. |
 * | `crossover_type`             | enum     | `CrossoverType`      | `ROULETTE`
 * (discrete) or `GEOMETRIC` (blend).         | | `num_independent_populations`|
 * unsigned | \f$\ge 1\f$          | Island-model populations evolved in
 * parallel.         | | `num_incumbent_solutions`    | unsigned | \f$\ge 0\f$
 * | Max non-dominated incumbents to retain (0 = adaptive).| | `pr_type` | enum
 * | `PathRelinking::Type`| Relinking strategy (ALLOCATION / PERMUTATION /
 * BINARY_SEARCH).| | `pr_percentage`              | double   | \f$(0, 1]\f$ |
 * Fraction of the path length to explore.               |
 *
 * \see NSBRKGA::BiasFunctionType
 * \see NSBRKGA::DiversityFunctionType
 * \see NSBRKGA::CrossoverType
 * \see NSBRKGA::PathRelinking::Type
 * \see NSBRKGA::readConfiguration()
 * \see NSBRKGA::NSBRKGA
 */
class NsbrkgaParams {
  public:
    /** \name NSBRKGA Hyper-parameters */
    //@{
    /// Number of elements in the population.  Must be \f$\ge 2\f$.
    unsigned population_size;

    /// Minimum fraction of `population_size` forming the elite set.
    /// Value in \f$(0, 1]\f$; a value of 0.1 means at least 10\% are elite.
    double min_elites_percentage;

    /// Maximum fraction of `population_size` for the elite set.
    /// Must be \f$\ge\f$ `min_elites_percentage` and in \f$(0, 1]\f$.
    double max_elites_percentage;

    /// Per-allele polynomial-mutation probability.  Value in \f$[0, 1]\f$;
    /// 0 disables mutation entirely.
    double mutation_probability;

    /// Polynomial-mutation distribution index \f$\eta_m > 0\f$.
    /// Larger values concentrate the perturbation near the original allele.
    double mutation_distribution;

    /// Number of **elite** parents selected per mating.
    /// Must satisfy \f$1 \le \texttt{num\_elite\_parents} \le
    /// \min(\texttt{total\_parents},\,\texttt{min\_num\_elites})\f$.
    unsigned num_elite_parents;

    /// Total parents per mating (elite + non-elite).
    /// Must be \f$\ge 2\f$ and \f$\ge \texttt{num\_elite\_parents}\f$.
    unsigned total_parents;

    /// Bias function preset controlling parent-selection probability.
    BiasFunctionType bias_type;

    /// Diversity function preset for adaptive elite-set sizing.
    DiversityFunctionType diversity_type;

    /// Crossover operator (ROULETTE or GEOMETRIC).
    CrossoverType crossover_type;

    /// Number of independent island-model populations evolved in parallel.
    /// Each island is evolved independently; elite migration is performed by
    /// `NSBRKGA::exchangeElite()`.
    unsigned num_independent_populations;

    /// Maximum number of non-dominated incumbent solutions to retain.
    /// Set to 0 to let the algorithm choose adaptively.
    unsigned num_incumbent_solutions;
    //@}

    /** \name Path Relinking parameters */
    //@{
    /// Path relinking strategy; see `PathRelinking::Type`.
    PathRelinking::Type pr_type;

    /// Fraction of the path length to explore during path relinking.
    /// Value in \f$(0, 1]\f$; 1.0 explores the entire path.
    double pr_percentage;
    //@}

  public:
    /** \name Default operators */
    //@{
    /// Default constructor.
    NsbrkgaParams()
        : population_size(0), min_elites_percentage(0.0),
          max_elites_percentage(0.0), mutation_probability(0.0),
          mutation_distribution(0.0), num_elite_parents(0), total_parents(0),
          bias_type(BiasFunctionType::CONSTANT),
          diversity_type(DiversityFunctionType::NONE),
          crossover_type(CrossoverType::ROULETTE),
          num_independent_populations(0), num_incumbent_solutions(0),
          pr_type(PathRelinking::Type::ALLOCATION), pr_percentage(0.0) {}

    /// Assignment operator for compliance.
    NsbrkgaParams &operator=(const NsbrkgaParams &) = default;

    /// Destructor.
    ~NsbrkgaParams() = default;
    //@}
};

//----------------------------------------------------------------------------//
// External Control Params class.
//----------------------------------------------------------------------------//

/**
 * \brief Additional control parameters consumed by the calling application.
 *
 * \details
 * These parameters govern the *outer loop* orchestration (e.g., when to call
 * `exchangeElite()`, `pathRelink()`, `shake()`, `reset()`) but are not used
 * directly by the NSBRKGA engine.  They are persisted to and from the same
 * configuration file as `NsbrkgaParams` for convenience.
 *
 * A value of **0** for any interval parameter disables the corresponding
 * operation.
 *
 * \see NSBRKGA::readConfiguration()
 * \see NSBRKGA::writeConfiguration()
 */
class ExternalControlParams {
  public:
    /// Number of generations between elite-chromosome exchanges across
    /// populations.  0 means no exchange is performed.
    unsigned exchange_interval;

    /// Number of elite chromosomes copied from each population during an
    /// exchange.  Meaningful only when `exchange_interval > 0`.
    unsigned num_exchange_individuals;

    /// Number of generations between path relinking calls.
    /// 0 means path relinking is never applied automatically.
    unsigned path_relink_interval;

    /// Number of generations between population-shaking calls.
    /// 0 means shaking is never applied automatically.
    unsigned shake_interval;

    /// Number of generations between full population resets.
    /// 0 means the population is never reset automatically.
    unsigned reset_interval;

  public:
    /** \name Default operators */
    //@{
    /// Default constructor.
    ExternalControlParams()
        : exchange_interval(0), num_exchange_individuals(0),
          path_relink_interval(0), shake_interval(0), reset_interval(0) {}

    /// Assignment operator for compliance.
    ExternalControlParams &operator=(const ExternalControlParams &) = default;

    /// Destructor.
    ~ExternalControlParams() = default;
    //@}
};

//----------------------------------------------------------------------------//
// Loading the parameters from file
//----------------------------------------------------------------------------//

/**
 * \brief Reads algorithm parameters from a configuration file.
 *
 * \details
 * The configuration file contains key-value pairs, one per line, in any
 * order.  Lines starting with `#` and blank lines are ignored.  All keys are
 * case-insensitive.  Every key in `NsbrkgaParams` (except `CROSSOVER_TYPE`,
 * which is optional for backward compatibility) and `ExternalControlParams`
 * must be present.
 *
 * \param filename path to the configuration file.
 * \return A pair `{NsbrkgaParams, ExternalControlParams}` populated from the
 *         file contents.
 * \throws std::fstream::failure if the file cannot be opened, contains an
 *         unknown token, a duplicate token, an unparseable value, or a
 *         required token is missing.
 * \see NSBRKGA::writeConfiguration()
 * \see NSBRKGA::NsbrkgaParams
 * \see NSBRKGA::ExternalControlParams
 */
INLINE std::pair<NsbrkgaParams, ExternalControlParams>
readConfiguration(const std::string &filename) {
    std::ifstream input(filename, std::ios::in);
    std::stringstream error_msg;

    if (!input) {
        error_msg << "File '" << filename << "' cannot be opened!";
        throw std::fstream::failure(error_msg.str());
    }

    std::unordered_map<std::string, bool> tokens({
        {"POPULATION_SIZE", false},
        {"MIN_ELITES_PERCENTAGE", false},
        {"MAX_ELITES_PERCENTAGE", false},
        {"MUTATION_PROBABILITY", false},
        {"MUTATION_DISTRIBUTION", false},
        {"NUM_ELITE_PARENTS", false},
        {"TOTAL_PARENTS", false},
        {"BIAS_TYPE", false},
        {"DIVERSITY_TYPE", false},
        {"CROSSOVER_TYPE", false},
        {"NUM_INDEPENDENT_POPULATIONS", false},
        {"NUM_INCUMBENT_SOLUTIONS", false},
        {"PR_TYPE", false},
        {"PR_PERCENTAGE", false},
        {"EXCHANGE_INTERVAL", false},
        {"NUM_EXCHANGE_INDIVIDUALS", false},
        {"PATH_RELINK_INTERVAL", false},
        {"SHAKE_INTERVAL", false},
        {"RESET_INTERVAL", false},
    });

    NsbrkgaParams nsbrkga_params;
    ExternalControlParams control_params;

    std::string line;
    unsigned line_count = 0;

    while (std::getline(input, line)) {
        line_count++;
        std::string::size_type pos = line.find_first_not_of(" \t\n\v");

        // Ignore all comments and blank lines.
        if (pos == std::string::npos || line[pos] == '#') {
            continue;
        }

        std::stringstream line_stream(line);
        std::string token, data;

        line_stream >> token >> data;

        std::transform(token.begin(), token.end(), token.begin(), toupper);
        if (tokens.find(token) == tokens.end()) {
            error_msg << "Invalid token on line " << line_count << ": "
                      << token;
            throw std::fstream::failure(error_msg.str());
        }

        if (tokens[token]) {
            error_msg << "Duplicate attribute on line " << line_count << ": "
                      << token << " already read!";
            throw std::fstream::failure(error_msg.str());
        }

        std::stringstream data_stream(data);
        bool fail = false;

        // TODO: for c++17, we may use std:any to short this code using a loop.
        if (token == "POPULATION_SIZE") {
            fail = !bool(data_stream >> nsbrkga_params.population_size);
        } else if (token == "MIN_ELITES_PERCENTAGE") {
            fail = !bool(data_stream >> nsbrkga_params.min_elites_percentage);
        } else if (token == "MAX_ELITES_PERCENTAGE") {
            fail = !bool(data_stream >> nsbrkga_params.max_elites_percentage);
        } else if (token == "MUTATION_PROBABILITY") {
            fail = !bool(data_stream >> nsbrkga_params.mutation_probability);
        } else if (token == "MUTATION_DISTRIBUTION") {
            fail = !bool(data_stream >> nsbrkga_params.mutation_distribution);
        } else if (token == "NUM_ELITE_PARENTS") {
            fail = !bool(data_stream >> nsbrkga_params.num_elite_parents);
        } else if (token == "TOTAL_PARENTS") {
            fail = !bool(data_stream >> nsbrkga_params.total_parents);
        } else if (token == "BIAS_TYPE") {
            fail = !bool(data_stream >> nsbrkga_params.bias_type);
        } else if (token == "DIVERSITY_TYPE") {
            fail = !bool(data_stream >> nsbrkga_params.diversity_type);
        } else if (token == "CROSSOVER_TYPE") {
            fail = !bool(data_stream >> nsbrkga_params.crossover_type);
        } else if (token == "NUM_INDEPENDENT_POPULATIONS") {
            fail = !bool(data_stream >>
                         nsbrkga_params.num_independent_populations);
        } else if (token == "NUM_INCUMBENT_SOLUTIONS") {
            fail = !bool(data_stream >> nsbrkga_params.num_incumbent_solutions);
        } else if (token == "PR_TYPE") {
            fail = !bool(data_stream >> nsbrkga_params.pr_type);
        } else if (token == "PR_PERCENTAGE") {
            fail = !bool(data_stream >> nsbrkga_params.pr_percentage);
        } else if (token == "EXCHANGE_INTERVAL") {
            fail = !bool(data_stream >> control_params.exchange_interval);
        } else if (token == "NUM_EXCHANGE_INDIVIDUALS") {
            fail =
                !bool(data_stream >> control_params.num_exchange_individuals);
        } else if (token == "PATH_RELINK_INTERVAL") {
            fail = !bool(data_stream >> control_params.path_relink_interval);
        } else if (token == "SHAKE_INTERVAL") {
            fail = !bool(data_stream >> control_params.shake_interval);
        } else if (token == "RESET_INTERVAL") {
            fail = !bool(data_stream >> control_params.reset_interval);
        }

        if (fail) {
            error_msg << "Invalid value for '" << token << "' on line "
                      << line_count << ": '" << data << "'";
            throw std::fstream::failure(error_msg.str());
        }

        tokens[token] = true;
    }

    for (const auto &attribute_flag : tokens) {
        // CROSSOVER_TYPE is optional for backward compatibility.
        if (!attribute_flag.second &&
            attribute_flag.first != "CROSSOVER_TYPE") {
            error_msg << "Argument '" << attribute_flag.first
                      << "' was not supplied in the config file";
            throw std::fstream::failure(error_msg.str());
        }
    }

    return std::make_pair(std::move(nsbrkga_params), std::move(control_params));
}

//----------------------------------------------------------------------------//
// Writing the parameters into file
//----------------------------------------------------------------------------//

/**
 * \brief Writes algorithm parameters to a configuration file.
 *
 * \details
 * The output file can be read back by `readConfiguration()`.  All parameters
 * from both `nsbrkga_params` and `control_params` are written, one per line,
 * in the format `KEY value`.
 *
 * \param filename path to the output file (created or truncated).
 * \param nsbrkga_params algorithm hyper-parameters to write.
 * \param control_params external control parameters to write;
 *        defaults to a zero-initialised object.
 * \throws std::fstream::failure if the file cannot be opened for writing.
 * \see NSBRKGA::readConfiguration()
 */
INLINE void writeConfiguration(
    const std::string &filename, const NsbrkgaParams &nsbrkga_params,
    const ExternalControlParams &control_params = ExternalControlParams()) {

    std::ofstream output(filename, std::ios::out);
    if (!output) {
        std::stringstream error_msg;
        error_msg << "File '" << filename << "' cannot be opened!";
        throw std::fstream::failure(error_msg.str());
    }

    output << "population_size " << nsbrkga_params.population_size << std::endl
           << "min_elites_percentage " << nsbrkga_params.min_elites_percentage
           << std::endl
           << "max_elites_percentage " << nsbrkga_params.max_elites_percentage
           << std::endl
           << "mutation_probability " << nsbrkga_params.mutation_probability
           << std::endl
           << "mutation_distribution " << nsbrkga_params.mutation_distribution
           << std::endl
           << "num_elite_parents " << nsbrkga_params.num_elite_parents
           << std::endl
           << "total_parents " << nsbrkga_params.total_parents << std::endl
           << "bias_type " << nsbrkga_params.bias_type << std::endl
           << "diversity_type " << nsbrkga_params.diversity_type << std::endl
           << "crossover_type " << nsbrkga_params.crossover_type << std::endl
           << "num_independent_populations "
           << nsbrkga_params.num_independent_populations << std::endl
           << "num_incumbent_solutions "
           << nsbrkga_params.num_incumbent_solutions << std::endl
           << "pr_type " << nsbrkga_params.pr_type << std::endl
           << "pr_percentage " << nsbrkga_params.pr_percentage << std::endl
           << "exchange_interval " << control_params.exchange_interval
           << std::endl
           << "num_exchange_individuals "
           << control_params.num_exchange_individuals << std::endl
           << "path_relink_interval " << control_params.path_relink_interval
           << std::endl
           << "shake_interval " << control_params.shake_interval << std::endl
           << "reset_interval " << control_params.reset_interval << std::endl;

    output.close();
}

//----------------------------------------------------------------------------//
// The Non-dominated Sorting Multi-Parent Biased Random-key Genetic Algorithm
//----------------------------------------------------------------------------//

/**
 * \brief Non-dominated Sorting Biased Random-Key Genetic Algorithm
 *        with Multi-Parent crossover and Implicit Path Relinking.
 *
 * \tparam Decoder A class or functor providing the decoding interface.
 *         It must expose:
 *         \code{.cpp}
 *         std::vector<double> decode(NSBRKGA::Chromosome &chromosome,
 *                                    bool rewrite);
 *         \endcode
 *         `decode()` maps a chromosome (random-key vector in \f$[0,1)^n\f$)
 *         to a vector of objective values.  When `rewrite == true` the decoder
 *         may overwrite the chromosome alleles (Lamarckian write-back); all
 *         modified alleles must remain in \f$[0,1)\f$.  When
 *         `rewrite == false` the chromosome must not be modified.
 *         **If `max_threads > 1`, `decode()` must be thread-safe.**
 *
 * \details
 *
 * ## Design and invariants
 *
 * ### Chromosome
 * Each chromosome is a `std::vector<double>` of length \f$n\f$ (the problem
 * dimension), with alleles in \f$[0, 1)\f$.  The library guarantees that
 * all generated alleles are initialised in \f$[0,1)\f$ and that polynomial
 * mutation keeps them in that range.
 *
 * ### Elite set
 * After each generation, non-dominated sorting assigns solutions to Pareto
 * fronts.  All front-0 solutions are elite; additional solutions from
 * subsequent fronts may be added using the diversity function until the
 * elite set size is in [min_num_elites, max_num_elites].
 *
 * ### Multi-objective handling
 * Optimization senses are given as `std::vector<Sense>`, one per objective.
 * Non-dominated sorting uses these senses to determine dominance: solution
 * \f$a\f$ dominates \f$b\f$ when \f$a\f$ is at least as good as \f$b\f$ in
 * every objective and strictly better in at least one.
 * Within each Pareto front, solutions are ordered by decreasing crowding
 * distance (NSGA-II strategy) to maintain diversity.
 *
 * ### Thread safety
 * Population decoding inside `evolve()` and `initialize()` is parallelised
 * via OpenMP.  The library guarantees that each call to `Decoder::decode()`
 * operates on a distinct chromosome object; however, any shared mutable state
 * in the decoder (e.g., problem data structures) must be protected by the
 * user.  A common safe pattern is to use only const/read-only shared data and
 * keep all working memory in local (stack-allocated) variables.
 *
 * ### Path relinking
 * Implicit Path Relinking (IPR) navigates from a base chromosome to a
 * guiding chromosome by iteratively selecting and applying the move that
 * brings the base closest to the guide.  Three strategies are available
 * (see `PathRelinking::Type`).  During relinking, `decode()` is called with
 * `rewrite = false` to preserve the search path.  A final call with
 * `rewrite = true` is issued on the best chromosome found.
 *
 * ## Minimal usage example
 *
 * \code{.cpp}
 * #include "nsbrkga/nsbrkga.hpp"
 *
 * // Problem-specific decoder.
 * struct MyDecoder {
 *     // Returns {objective_value}.
 *     std::vector<double> decode(NSBRKGA::Chromosome &chr, bool) const {
 *         double cost = 0.0;
 *         for (double allele : chr) { cost += allele; }  // trivial example
 *         return {cost};
 *     }
 * };
 *
 * int main() {
 *     auto [params, ctrl] =
 *         NSBRKGA::readConfiguration("brkga.cfg");
 *
 *     MyDecoder decoder;
 *     NSBRKGA::NSBRKGA<MyDecoder> algo(
 *         decoder,
 *         {NSBRKGA::Sense::MINIMIZE},  // one objective, minimise
 *         42,                          // RNG seed
 *         100,                         // chromosome size
 *         params);
 *
 *     algo.initialize();
 *
 *     for (unsigned gen = 0; gen < 200; ++gen) {
 *         algo.evolve();  // one generation
 *
 *         if (ctrl.exchange_interval > 0 &&
 *             gen % ctrl.exchange_interval == 0) {
 *             algo.exchangeElite(ctrl.num_exchange_individuals);
 *         }
 *     }
 *
 *     for (auto &[fitness, chrom] : algo.getIncumbentSolutions()) {
 *         std::cout << "fitness: " << fitness[0] << "\n";
 *     }
 * }
 * \endcode
 *
 * ## History
 * Based on the original BRKGA code by Rodrigo Franco Toso (2011).
 * http://github.com/rfrancotoso/brkgaAPI
 *
 * \see NSBRKGA::Chromosome
 * \see NSBRKGA::NsbrkgaParams
 * \see NSBRKGA::ExternalControlParams
 * \see NSBRKGA::BiasFunctionType
 * \see NSBRKGA::CrossoverType
 * \see NSBRKGA::PathRelinking::Type
 * \see NSBRKGA::DistanceFunctionBase
 */
template <class Decoder> class NSBRKGA {
  public:
    /** \name Constructors and destructor */
    //@{
    /**
     * \brief Constructs the algorithm and allocates internal data structures.
     *
     * \details
     * All data in `params` is copied; the caller may discard or modify the
     * object after construction.  The constructor validates all parameter
     * combinations and throws `std::range_error` on any violation.
     *
     * \param decoder_reference reference to the user-supplied decoder object.
     *        The algorithm stores a reference — the decoder must outlive the
     *        `NSBRKGA` instance.
     * \param senses vector of optimization senses, one per objective.
     *        E.g. `{Sense::MINIMIZE, Sense::MAXIMIZE}` for a bi-objective
     *        problem where the first objective is minimised and the second
     *        is maximised.
     * \param seed seed for the Mersenne Twister RNG.
     * \param chromosome_size number of alleles \f$n\f$ in each chromosome;
     *        must be \f$\ge 1\f$.
     * \param params algorithm hyper-parameters; see `NsbrkgaParams`.
     * \param max_threads number of OpenMP threads for parallel decoding.
     *        **`Decoder::decode()` must be thread-safe when > 1.**
     *        Default: 1 (serial).
     * \param evolutionary_mechanism_on when `false`, the elite set is forced
     *        to size 1 and all other individuals are mutants, effectively
     *        turning the algorithm into a multi-start random restart.  Useful
     *        for benchmarking decoders.  Default: `true`.
     *
     * \throws std::range_error if any parameter value or combination is
     *         invalid (e.g., zero chromosome size, zero population, elite
     *         bounds crossed, invalid parent counts).
     *
     * \see NSBRKGA::NsbrkgaParams
     * \see NSBRKGA::Sense
     */
    NSBRKGA(Decoder &decoder_reference, const std::vector<Sense> senses,
            const unsigned seed, const unsigned chromosome_size,
            const NsbrkgaParams &params, const unsigned max_threads = 1,
            const bool evolutionary_mechanism_on = true);

    /// Destructor.
    ~NSBRKGA() {}
    //@}

    /** \name Initialization methods */
    //@{
    /**
     * \brief Sets individuals to initial population.
     *
     * Set initial individuals into the population to work as warm-starters.
     * Such individuals can be obtained from solutions of external procedures
     * such as fast heuristics, other metaheuristics, or even relaxations from
     * a mixed integer programming model that models the problem.
     *
     * All given solutions are assigned to one population only. Therefore, the
     * maximum number of solutions is the size of the populations.
     *
     * \param populations a set of individuals encoded as Chromosomes.
     * \throw std::runtime_error if the number of given chromosomes is larger
     *        than the population size; if the sizes of the given chromosomes
     *        do not match with the required chromosome size.
     */
    void setInitialPopulations(
        const std::vector<std::vector<Chromosome>> &populations);

    /**
     * \brief Sets a custom bias function used to build the probabilities.
     *
     * It must be a **positive non-increasing function**, i.e.
     * \f$ f: \mathbb{N}^+ \to \mathbb{R}^+\f$ such that
     * \f$f(i) \ge 0\f$ and \f$f(i) \ge f(i+1)\f$ for
     * \f$i \in [1..total\_parents]\f$.
     * For example
     * \code{.cpp}
     *      setBiasCustomFunction(
     *          [](const unsigned x) {
     *              return 1.0 / (x * x);
     *          }
     *      );
     * \endcode
     * sets an inverse quadratic function.
     *
     * \param func a reference to a unary positive non-increasing function.
     * \throw std::runtime_error in case the function is not a non-decreasing
     *        positive function.
     */
    void
    setBiasCustomFunction(const std::function<double(const unsigned)> &func);

    /**
     * \brief Sets a custom diversity function used to build the elite set.
     */
    void setDiversityCustomFunction(
        const std::function<double(const std::vector<std::vector<double>> &)>
            &func);

    /**
     * \brief Initializes the populations and others parameters of the
     *        algorithm.
     *
     * If a initial population is supplied, this method completes the remain
     * individuals, if they do not exist. This method also performs the initial
     * decoding of the chromosomes. Therefore, depending on the decoder
     * implementation, this can take a while, and the user may want to time
     * such procedure in his/her experiments.
     *
     * \warning
     *      This method must be call before any evolutionary or population
     *      handling method.
     *
     * \warning
     *     As it is in #evolve(), the decoding is done in parallel using
     *     threads, and the user **must guarantee that the decoder is
     *     THREAD-SAFE.** If such property cannot be held, we suggest using
     *     a single thread  for optimization.
     */
    void initialize();
    //@}

    /** \name Evolution */
    //@{
    /**
     * \brief Evolves the current populations following the guidelines of
     *        NSBRKGAs.
     *
     * \warning
     *     The decoding is done in parallel using threads, and the user **must
     *     guarantee that the decoder is THREAD-SAFE.** If such property cannot
     *     be held, we suggest using a single thread for optimization.
     *
     * \param generations number of generations to be evolved. Must be larger
     *        than zero.
     * \throw std::runtime_error if the algorithm is not initialized.
     * \throw std::range_error if the number of generations is zero.
     */
    bool evolve(unsigned generations = 1);
    //@}

    /** \name Path relinking */
    //@{
    /**
     * \brief Performs path relinking between elite solutions that are, at
     * least, a given minimum distance between themselves. In this method, the
     * local/loaded parameters are ignored in favor to the supplied ones.
     *
     * In the presence of multiple populations, the path relinking is performed
     * between elite chromosomes from different populations, in a circular
     * fashion. For example, suppose we have 3 populations. The framework
     * performs 3 path relinkings: the first between individuals from
     * populations 1 and 2, the second between populations 2 and 3, and the
     * third between populations 3 and 1. In the case of just one population,
     * both base and guiding individuals are sampled from the elite set of that
     * population.
     *
     * Note that the algorithm tries to find a pair of base and guiding
     * solutions with a minimum distance given by the distance function. If this
     * is not possible, a new pair of solutions are sampled (without
     * replacement) and tested against the distance. In case it is not possible
     * to find such pairs for the given populations, the algorithm skips to the
     * next pair of populations (in a circular fashion, as described above).
     * Yet, if such pairs are not found in any case, the algorithm declares
     * failure. This indicates that the populations are very homogeneous.
     *
     * If the found solution is the best solution found so far, IPR replaces the
     * worst solution by it. Otherwise, IPR computes the distance between the
     * found solution and all other solutions in the elite set, and replaces the
     * worst solution by it if and only if the found solution is, at least,
     * `minimum_distance` from all them.
     *
     * The API will call `Decoder::decode()` function always with `rewrite =
     * false`. The reason is that if the decoder rewrites the chromosome, the
     * path between solutions is lost and inadvertent results may come up. Note
     * that at the end of the path relinking, the method calls the decoder with
     * `rewrite = true` in the best chromosome found to guarantee that this
     * chromosome is re-written to reflect the best solution found.
     *
     * This method is a multi-thread implementation. Instead of to build and
     * decode each chromosome one at a time, the method builds a list of
     * candidates, altering the alleles/keys according to the guide solution,
     * and then decode all candidates in parallel. Note that
     * `O(chromosome_size^2 / block_size)` additional memory is necessary to
     * build the candidates, which can be costly if the `chromosome_size` is
     * very large.
     *
     * \warning
     *     As it is in #evolve(), the decoding is done in parallel using
     *     threads, and the user **must guarantee that the decoder is
     *     THREAD-SAFE.** If such property cannot be held, we suggest using
     *     a single thread  for optimization.
     *
     * \param pr_type type of path relinking to be performed.
     *        See PathRelinking::Type.
     * \param dist a pointer to a functor/object to compute the distance between
     *        two chromosomes. This object must be inherited from
     *        NSBRKGA::DistanceFunctionBase and implement its methods.
     * \param max_time aborts path relinking when reach `max_time`.
     *        If `max_time <= 0`, no limit is imposed.
     *        Default: 0 (no limit).
     * \param percentage defines the size, in percentage, of the path to build.
     *        Default: 1.0 (100%).
     *
     * \returns A PathRelinking::PathRelinkingResult depending on the relink
     *          status.
     *
     * \throw std::range_error if the percentage or size of the path is
     *        not in (0, 1].
     */
    PathRelinking::PathRelinkingResult
    pathRelink(PathRelinking::Type pr_type,
               std::shared_ptr<DistanceFunctionBase> dist, long max_time = 0,
               double percentage = 1.0);

    /**
     * \brief Performs path relinking between elite solutions that are,
     *        at least, a given minimum distance between themselves.
     *
     * This method uses all parameters supplied in the constructor.
     * In particular, the block size is computed by
     * \f$\lceil \alpha \times \sqrt{p} \rceil\f$
     * where \f$\alpha\f$ is NsbrkgaParams#alpha_block_size and
     * \f$p\f$ is NsbrkgaParams#population_size.
     * If the size is larger than the chromosome size, the size is set to
     * half of the chromosome size.
     *
     * Please, refer to #pathRelink() for details.
     *
     * \param dist a pointer to a functor/object to compute the distance between
     *        two chromosomes. This object must be inherited from
     *        NSBRKGA::DistanceFunctionBase and implement its methods.
     * \param max_time aborts path relinking when reach `max_time`.
     *        If `max_time <= 0`, no limit is imposed.
     *        Default: 0 (no limit).
     * \param percentage defines the size, in percentage, of the path to build.
     *        Default: 1.0 (100%).
     *
     * \returns A PathRelinking::PathRelinkingResult depending on the relink
     *          status.
     *
     * \throw std::range_error if the percentage or size of the path is
     *        not in (0, 1].
     */
    PathRelinking::PathRelinkingResult
    pathRelink(std::shared_ptr<DistanceFunctionBase> dist, long max_time = 0,
               double percentage = 1.0);
    //@}

    /** \name Population manipulation methods */
    //@{
    /**
     * \brief Exchanges elite-solutions between the populations.
     *
     * Given a population, the `num_immigrants` best solutions are copied to
     * the neighbor populations, replacing their worth solutions. If there is
     * only one population, nothing is done.
     *
     * \param num_immigrants number of elite chromosomes to select from each
     *      population.
     * \throw std::range_error if the number of immigrants less than one or
     *        it is larger than or equal to the population size divided by
     *        the number of populations minus one, i.e. \f$\lceil
     *        \frac{population\_size}{num\_independent\_populations} \rceil
     *         - 1\f$.
     */
    void exchangeElite(unsigned num_immigrants);

    /**
     * \brief Resets all populations with brand new keys.
     *
     * All warm-start solutions provided setInitialPopulation() are discarded.
     * You may use injectChromosome() to insert those solutions again.
     * \param intensity the intensity of the reset.
     * \param population_index the index of the population to be reset. If
     * `population_index >= num_independent_populations`, all populations
     * are reset.
     * \throw std::runtime_error if the algorithm is not initialized.
     */
    void reset(
        double intensity = 0.5,
        unsigned population_index = std::numeric_limits<unsigned>::infinity());

    /**
     * \brief Performs a shaking in the chosen population.
     * \param intensity the intensity of the shaking.
     * \param distribution the distribution of the shaking.
     * \param population_index the index of the population to be shaken. If
     * `population_index >= num_independent_populations`, all populations
     * are shaken.
     * \throw std::runtime_error if the algorithm is not initialized.
     */
    void shake(
        double intensity = 0.5, double distribution = 20.0,
        unsigned population_index = std::numeric_limits<unsigned>::infinity());

    /**
     * \brief Injects a chromosome and its fitness into a population in the
     *         given place position.
     *
     * If fitness is not provided (`fitness == Inf`), the decoding is performed
     * over chromosome. Once the chromosome is injected, the population is
     * re-sorted according to the chromosomes' fitness.
     *
     * \param chromosome the chromosome to be injected.
     * \param population_index the population index.
     * \param position the chromosome position.
     *
     * \throw std::range_error either if `population_index` is larger
     *        than number of populations; or `position` is larger than the
     *        population size; or ` chromosome.size() != chromosome_size`
     */
    void injectChromosome(const Chromosome &chromosome,
                          unsigned population_index, unsigned position);
    //@}

    /** \name Support methods */
    //@{
    /**
     * \brief Returns a reference to a current population.
     * \param population_index the population index.
     * \throw std::range_error if the index is larger than number of
     *        populations.
     */
    const Population &getCurrentPopulation(unsigned population_index = 0) const;

    /// Returns the solutions with non-dominated fitness found so far.
    const std::vector<std::pair<std::vector<double>, Chromosome>> &
    getIncumbentSolutions() const;

    /// Returns the chromosomes with non-dominated fitness found so far.
    std::vector<Chromosome> getIncumbentChromosomes() const;

    /// Return the non-dominated fitness found so far.
    std::vector<std::vector<double>> getIncumbentFitnesses() const;

    /**
     * \brief Returns a reference to a chromosome of the given population.
     * \param population_index the population index.
     * \param position the chromosome position, ordered by fitness.
     *        The best chromosome is located in position 0.
     * \throw std::range_error either if `population_index` is larger
     *        than number of populations, or `position` is larger than the
     *        population size.
     */
    const Chromosome &getChromosome(unsigned population_index,
                                    unsigned position) const;

    /**
     * \brief Returns the fitness of a chromosome of the given population.
     * \param population_index the population index.
     * \param position the chromosome position, ordered by fitness.
     *        The best chromosome is located in position 0.
     * \throw std::range_error either if `population_index` is larger
     *        than number of populations, or `position` is larger than the
     *        population size.
     */
    std::vector<double> getFitness(unsigned population_index,
                                   unsigned position) const;
    //@}

    /** \name Parameter getters */
    //@{
    const NsbrkgaParams &getNsbrkgaParams() const { return this->params; }

    std::vector<Sense> getOptimizationSenses() const {
        return this->OPT_SENSES;
    }

    unsigned getChromosomeSize() const { return this->CHROMOSOME_SIZE; }

    unsigned getMinNumElites() const { return this->min_num_elites; }

    unsigned getMaxNumElites() const { return this->max_num_elites; }

    bool evolutionaryIsMechanismOn() const {
        return this->evolutionary_mechanism_on;
    }

    unsigned getMaxThreads() const { return this->MAX_THREADS; }
    //@}

  protected:
    /** \name BRKGA Hyper-parameters */
    //@{
    /// The BRKGA and IPR hyper-parameters.
    NsbrkgaParams params;

    /// Indicates whether we are maximizing or minimizing each objective.
    const std::vector<Sense> OPT_SENSES;

    /// Number of genes in the chromosome.
    const unsigned CHROMOSOME_SIZE;

    /// Minimum number of elite individuals in each population.
    unsigned min_num_elites;

    /// Maximum number of elite individuals in each population.
    unsigned max_num_elites;

    /// If false, no evolution is performed but only chromosome decoding.
    /// Very useful to emulate a multi-start algorithm.
    bool evolutionary_mechanism_on;
    //@}

    /** \name Parallel computing parameters */
    //@{
    /// Number of threads for parallel decoding.
    const unsigned MAX_THREADS;
    //@}

  protected:
    /** \name Engines */
    //@{
    /// Reference to the problem-dependent Decoder.
    Decoder &decoder;

    /// Mersenne twister random number generator.
    std::mt19937 rng;
    //@}

    /** \name Algorithm data */
    //@{
    /// Previous populations.
    std::vector<std::shared_ptr<Population>> previous;

    /// Current populations.
    std::vector<std::shared_ptr<Population>> current;

    /// Reference for the bias function.
    std::function<double(const unsigned)> bias_function;

    /// Reference for the diversity function.
    std::function<double(const std::vector<std::vector<double>> &)>
        diversity_function;

    /// Holds the sum of the results of each raking given a bias function.
    /// This value is needed to normalization.
    double total_bias_weight;

    /// Used to shuffled individual/chromosome indices during the selection.
    std::vector<unsigned> shuffled_individuals;

    /// Used to select the parents for the mating.
    std::vector<unsigned> parents_indexes;

    /// Defines the order of parents during the mating.
    std::vector<std::pair<std::vector<double>, unsigned>> parents_ordered;

    /// Indicates if initial populations are set.
    bool initial_populations;

    /// Indicates if the algorithm was proper initialized.
    bool initialized;

    /// Holds the start time for a call of the path relink procedure.
    std::chrono::system_clock::time_point pr_start_time;

    /// The current best solutions.
    std::vector<std::pair<std::vector<double>, Chromosome>> incumbent_solutions;
    //@}

  private:
    void selectParents(const Population &curr, const size_t &chr,
                       const bool use_best_individual = false);

    /**
     * @brief Applies polynomial mutation to a single allele value.
     *
     * Performs polynomial mutation on the given allele, a variation operator
     * commonly used in multi-objective evolutionary algorithms (e.g., NSGA-II).
     * The mutation perturbs the allele value using a polynomial probability
     * distribution, where the shape of the distribution is controlled by the
     * mutation distribution index.
     *
     * The mutation is applied probabilistically based on the given mutation
     * probability. When triggered, a random perturbation (delta) is computed
     * using the polynomial distribution, which favors small perturbations
     * near the original value while still allowing larger changes.
     *
     * The resulting allele value is clamped to the valid range [0.0, 1.0),
     * where the upper bound is set to RAND_MAX / (RAND_MAX + 1.0) to ensure
     * the allele remains strictly less than 1.0.
     *
     * @param[in,out] allele The allele value to be mutated, expected to be
     *                       in the range [0.0, 1.0). Modified in place if
     *                       mutation is applied.
     * @param[in] mutation_probability The probability of applying the mutation
     *                                 to the allele (value in [0.0, 1.0]).
     * @param[in] mutation_distribution The distribution index (eta_m) that
     *                                  controls the shape of the polynomial
     *                                  distribution. Higher values produce
     *                                  perturbations closer to the original
     *                                  value. Note: this parameter is unused
     *                                  in the body; the member variable
     *                                  `this->params.mutation_distribution`
     *                                  is used instead.
     */
    void polynomialMutation(double &allele, double mutation_probability,
                            double mutation_distribution);

    void polynomialMutation(double &allele, double mutation_probability);

    void polynomialMutation(double &allele);

    void mate(const Population &curr, Chromosome &offspring);

  protected:
    /** \name Core local methods */
    //@{
    /**
     * \brief Evolves the current population to the next.
     *
     * Note that the next population will be re-populate completely.
     *
     * \param[in] curr current population.
     * \param[out] next next population.
     */
    bool evolution(Population &curr, Population &next);

    /**
     * \brief Performs the direct path relinking.
     *
     * This method changes each allele or block of alleles of base chromosome
     * for the correspondent one in the guide chromosome.
     *
     * This method is a multi-thread implementation. Instead of to build and
     * decode each chromosome one at a time, the method builds a list of
     * candidates, altering the alleles/keys according to the guide solution,
     * and then decode all candidates in parallel. Note that
     * `O(chromosome_size^2)` additional memory is necessary to
     * build the candidates, which can be costly if the `chromosome_size` is
     * very large.
     *
     * \param solution1 first solution (fitness–chromosome pair).
     * \param solution2 second solution (fitness–chromosome pair).
     * \param max_time abort path relinking when reach `max_time`.
     *        If `max_time <= 0`, no limit is imposed.
     * \param percentage define the size, in percentage, of the path to build.
     * \param[out] best_solutions the best solutions found in the search.
     * \return the best solution found in the search.
     */
    std::pair<std::vector<double>, Chromosome> allocationPathRelink(
        const std::pair<std::vector<double>, Chromosome> &solution1,
        const std::pair<std::vector<double>, Chromosome> &solution2,
        long max_time, double percentage,
        std::vector<std::pair<std::vector<double>, Chromosome>>
            &best_solutions);

    /**
     * \brief Performs the permutation-based path relinking.
     *
     * In this method, the permutation induced by the keys in the guide
     * solution is used to change the order of the keys in the permutation
     * induced by the base solution.
     *
     * This method is a multi-thread implementation. Instead of to build and
     * decode each chromosome one at a time, the method builds a list of
     * candidates, altering the alleles/keys according to the guide solution,
     * and then decode all candidates in parallel. Note that
     * `O(chromosome_size^2)` additional memory is necessary to
     * build the candidates, which can be costly if the `chromosome_size` is
     * very large.
     *
     * The path relinking is performed by changing the order of
     * each allele of base chromosome for the correspondent one in
     * the guide chromosome.
     * \param solution1 first solution (fitness–chromosome pair).
     * \param solution2 second solution (fitness–chromosome pair).
     * \param max_time abort path relinking when reach `max_time`.
     *        If `max_time <= 0`, no limit is imposed.
     * \param percentage define the size, in percentage, of the path to build.
     * \param[out] best_solutions the best solutions found in the search.
     * \return the best solution found in the search.
     */
    std::pair<std::vector<double>, Chromosome> permutationPathRelink(
        const std::pair<std::vector<double>, Chromosome> &solution1,
        const std::pair<std::vector<double>, Chromosome> &solution2,
        long max_time, double percentage,
        std::vector<std::pair<std::vector<double>, Chromosome>>
            &best_solutions);

    /**
     * \brief Performs the binary-search-based path relinking.
     *
     * \param solution1 first solution (fitness–chromosome pair).
     * \param solution2 second solution (fitness–chromosome pair).
     * \param max_time abort path relinking when reach `max_time`.
     *        If `max_time <= 0`, no limit is imposed.
     * \param[out] best_solutions the best solutions found in the search.
     * \return the best solution found in the search.
     */
    std::pair<std::vector<double>, Chromosome> binarySearchPathRelink(
        const std::pair<std::vector<double>, Chromosome> &solution1,
        const std::pair<std::vector<double>, Chromosome> &solution2,
        long max_time,
        std::vector<std::pair<std::vector<double>, Chromosome>>
            &best_solutions);

    static bool updateIncumbentSolutions(
        std::vector<std::pair<std::vector<double>, Chromosome>>
            &incumbent_solutions,
        const std::vector<std::pair<std::vector<double>, Chromosome>>
            &new_solutions,
        const std::vector<Sense> &senses,
        const std::size_t max_num_solutions = 0);

    /**
     * @brief Updates the incumbent (best known) solutions with new candidate
     * solutions.
     *
     * This is a convenience member function that delegates to the static
     * version of updateIncumbentSolutions, passing the instance's incumbent
     * solutions, optimization senses, and the configured maximum number of
     * incumbent solutions.
     *
     * @tparam Decoder The decoder class used to evaluate chromosomes.
     *
     * @param new_solutions A vector of pairs, where each pair contains an
     * objective values vector and its corresponding chromosome, representing
     * the new candidate solutions to be considered for inclusion in the
     * incumbent set.
     *
     * @return True if the incumbent solutions were updated (i.e., at least one
     * new solution was added or replaced an existing one); false otherwise.
     */
    bool updateIncumbentSolutions(
        const std::vector<std::pair<std::vector<double>, Chromosome>>
            &newSolutions);
    //@}

    /** \name Helper functions */
    //@{
    /**
     * \brief Returns `true` if `a1` dominates `a2`.
     */
    inline bool dominates(const std::vector<double> a1,
                          const std::vector<double> a2) const;

    /**
     * @brief Generates a random double-precision floating-point number
     *        uniformly distributed in the range [0, 1).
     *
     * Uses std::generate_canonical to produce a high-precision random value
     * with as many random bits as the double type supports. This method is
     * preferred over manual scaling of the RNG output to avoid precision
     * issues that can occur on certain platforms (e.g., Linux).
     *
     * @return A uniformly distributed random double in the interval [0, 1).
     */
    inline double rand01();

    /**
     * @brief Generates a uniformly distributed random integer in the range [0,
     * n].
     *
     * Uses a rejection sampling technique adapted from Magnus Jonsson
     * (magnus@smartelectronix.com) to produce an unbiased random integer.
     * The method works by first computing a bitmask that covers all bits
     * used by the value @p n, then repeatedly drawing random values masked
     * to those bits until the result falls within the desired range [0, n].
     * This avoids the modulo bias that would result from a simple
     * modulus operation.
     *
     * @tparam Decoder The decoder type used by the NSBRKGA framework.
     * @param n The upper bound (inclusive) of the random integer range.
     *          Must be a non-negative value representable by uint_fast32_t.
     * @return A uniformly distributed random integer in the range [0, n].
     *
     * @note This method is specific to uint_fast32_t types (up to 32-bit
     * values).
     * @note The expected number of iterations is at most 2, since the mask
     *       ensures at least half of the drawn values fall within [0, n].
     */
    inline uint_fast32_t randInt(const uint_fast32_t n);
    //@}
};

//----------------------------------------------------------------------------//

template <class Decoder>
NSBRKGA<Decoder>::NSBRKGA(Decoder &_decoder_reference,
                          const std::vector<Sense> _senses, unsigned _seed,
                          unsigned _chromosome_size,
                          const NsbrkgaParams &_params,
                          const unsigned _max_threads,
                          const bool _evolutionary_mechanism_on)
    :

      // Algorithm parameters.
      params(_params), OPT_SENSES(_senses), CHROMOSOME_SIZE(_chromosome_size),
      min_num_elites(
          _evolutionary_mechanism_on
              ? unsigned(params.min_elites_percentage * params.population_size)
              : 1),
      max_num_elites(
          _evolutionary_mechanism_on
              ? unsigned(params.max_elites_percentage * params.population_size)
              : 1),
      evolutionary_mechanism_on(_evolutionary_mechanism_on),
      MAX_THREADS(_max_threads),

      // Internal data.
      decoder(_decoder_reference), rng(_seed),
      previous(params.num_independent_populations, nullptr),
      current(params.num_independent_populations, nullptr), bias_function(),
      diversity_function(), total_bias_weight(0.0),
      shuffled_individuals(params.population_size),
      parents_indexes(params.total_parents),
      parents_ordered(params.total_parents), initial_populations(false),
      initialized(false), pr_start_time(), incumbent_solutions() {
    using std::range_error;
    std::stringstream ss;

    if (this->CHROMOSOME_SIZE == 0) {
        ss << "Chromosome size must be larger than zero: "
           << this->CHROMOSOME_SIZE;
    } else if (this->params.population_size == 0) {
        ss << "Population size must be larger than zero: "
           << this->params.population_size;
    } else if (this->min_num_elites > this->max_num_elites) {
        ss << "Minimum elite-set size (" << this->min_num_elites
           << ") greater than maximum elite-set size (" << this->max_num_elites
           << ")";
    } else if (this->min_num_elites == 0) {
        ss << "Minimum elite-set size equals zero.";
    } else if (this->params.mutation_probability < 0) {
        ss << "Mutation probability (" << this->params.mutation_probability
           << ") smaller than zero.";
    } else if (this->params.mutation_distribution <=
               std::numeric_limits<double>::epsilon()) {
        ss << "Mutation distribution (" << this->params.mutation_distribution
           << ") smaller or equal to zero.";
    } else if (this->max_num_elites > this->params.population_size) {
        ss << "Maximum elite-set size (" << this->max_num_elites
           << ") greater than population size (" << this->params.population_size
           << ")";
    } else if (this->params.num_elite_parents < 1) {
        ss << "num_elite_parents must be at least 1: "
           << this->params.num_elite_parents;
    } else if (this->params.total_parents < 2) {
        ss << "Total_parents must be at least 2: "
           << this->params.total_parents;
    } else if (this->params.num_elite_parents > this->params.total_parents) {
        ss << "Num_elite_parents (" << this->params.num_elite_parents << ") "
           << "is greater than total_parents (" << this->params.total_parents
           << ")";
    } else if (this->params.num_elite_parents > this->min_num_elites) {
        ss << "Num_elite_parents (" << this->params.num_elite_parents
           << ") is greater than minimum elite-set size ("
           << this->min_num_elites << ")";
    } else if (this->params.num_independent_populations == 0) {
        ss << "Number of parallel populations cannot be zero.";
    } else if (this->params.pr_percentage < 1e-6 ||
               this->params.pr_percentage > 1.0) {
        ss << "Path relinking percentage (" << this->params.pr_percentage
           << ") is not in the range (0, 1].";
    }

    const auto str_error = ss.str();
    if (str_error.length() > 0) {
        throw range_error(str_error);
    }

    // Chooses the bias function.
    switch (this->params.bias_type) {
    case BiasFunctionType::LOGINVERSE: {
        // Same as log(r + 1), but avoids precision loss.
        this->setBiasCustomFunction(
            [](const unsigned r) { return 1.0 / log1p(r); });
        break;
    }

    case BiasFunctionType::LINEAR: {
        this->setBiasCustomFunction([](const unsigned r) { return 1.0 / r; });
        break;
    }

    case BiasFunctionType::QUADRATIC: {
        this->setBiasCustomFunction(
            [](const unsigned r) { return 1.0 / (r * r); });
        break;
    }

    case BiasFunctionType::CUBIC: {
        this->setBiasCustomFunction(
            [](const unsigned r) { return 1.0 / (r * r * r); });
        break;
    }

    case BiasFunctionType::EXPONENTIAL: {
        this->setBiasCustomFunction(
            [](const unsigned r) { return exp(-1.0 * r); });
        break;
    }

    case BiasFunctionType::SQRT: {
        this->setBiasCustomFunction(
            [](const unsigned r) { return 1.0 / sqrt(r); });
        break;
    }

    case BiasFunctionType::CBRT: {
        this->setBiasCustomFunction(
            [](const unsigned r) { return 1.0 / cbrt(r); });
        break;
    }

    case BiasFunctionType::CONSTANT:
    default: {
        this->setBiasCustomFunction(
            [&](const unsigned) { return 1.0 / this->params.total_parents; });
        break;
    }
    }

    // Chooses the diversity function.
    switch (this->params.diversity_type) {
    case DiversityFunctionType::NONE: {
        this->setDiversityCustomFunction(
            [](const std::vector<std::vector<double>> & /* not used */) {
                return 0.0;
            });
        break;
    }

    case DiversityFunctionType::AVERAGE_DISTANCE_BETWEEN_ALL_PAIRS: {
        this->setDiversityCustomFunction(
            [](const std::vector<std::vector<double>> &x) {
                double diversity = 0.0;

                if (x.size() < 2 || x.front().empty()) {
                    return diversity;
                }

                for (std::size_t i = 0; i + 1 < x.size(); i++) {
                    for (std::size_t j = i + 1; j < x.size(); j++) {
                        diversity += std::sqrt(std::inner_product(
                            x[i].begin(), x[i].end(), x[j].begin(), 0.0,
                            std::plus<>(), [](double a, double b) {
                                return (a - b) * (a - b);
                            }));
                    }
                }

                diversity /= (double)(x.size() * (x.size() - 1.0)) / 2.0;

                return diversity;
            });
        break;
    }

    case DiversityFunctionType::POWER_MEAN_BASED: {
        this->setDiversityCustomFunction(
            [](const std::vector<std::vector<double>> &x) {
                double diversity = 0.0;

                if (x.size() < 2 || x.front().empty()) {
                    return diversity;
                }

                for (const std::vector<double> &vec_i : x) {
                    double dist = 0.0;

                    for (const std::vector<double> &vec_j : x) {
                        double norm = std::numeric_limits<double>::max();

                        for (std::size_t k = 0;
                             k < vec_i.size() && k < vec_j.size(); k++) {
                            double delta = std::abs(vec_i[k] - vec_j[k]);

                            if (norm > delta) {
                                norm = delta;
                            }
                        }
                        dist += norm;
                    }

                    dist /= (double)(x.size() - 1.0);
                    diversity += dist;
                }

                diversity /= (double)x.size();

                return diversity;
            });
        break;
    }

    case DiversityFunctionType::AVERAGE_DISTANCE_TO_CENTROID:
    default: {
        this->setDiversityCustomFunction(
            [](const std::vector<std::vector<double>> &x) {
                double diversity = 0.0;

                if (x.size() < 2 || x.front().empty()) {
                    return diversity;
                }

                std::vector<double> centroid(x.front().size(), 0.0);

                for (const std::vector<double> &vec : x) {
                    std::transform(centroid.begin(), centroid.end(),
                                   vec.begin(), centroid.begin(),
                                   std::plus<>());
                }

                for (double &val : centroid) {
                    val /= (double)x.size();
                }

                for (const std::vector<double> &vec : x) {
                    diversity += std::sqrt(std::inner_product(
                        centroid.begin(), centroid.end(), vec.begin(), 0.0,
                        std::plus<>(),
                        [](double a, double b) { return (a - b) * (a - b); }));
                }
                diversity /= (double)x.size();

                return diversity;
            });
        break;
    }
    }

    this->rng.discard(1000); // Discard some states to warm up.
}

//----------------------------------------------------------------------------//

template <class Decoder>
inline bool NSBRKGA<Decoder>::dominates(const std::vector<double> a1,
                                        const std::vector<double> a2) const {
    return Population::dominates(a1, a2, this->OPT_SENSES);
}

//----------------------------------------------------------------------------//

/**
 * \brief Returns a reference to a current population.
 * \param population_index the population index.
 * \throw std::range_error if the index is larger than number of
 *        populations.
 */
template <class Decoder>
const Population &
NSBRKGA<Decoder>::getCurrentPopulation(unsigned population_index) const {
    if (population_index >= this->current.size()) {
        throw std::range_error("The index is larger than number of "
                               "populations");
    }
    return (*(this->current)[population_index]);
}

//----------------------------------------------------------------------------//

/// Return the non-dominated fitness found so far.
template <class Decoder>
std::vector<std::vector<double>>
NSBRKGA<Decoder>::getIncumbentFitnesses() const {
    std::vector<std::vector<double>> result;

    for (std::size_t i = 0; i < this->incumbent_solutions.size(); i++) {
        result.push_back(this->incumbent_solutions[i].first);
    }

    return result;
}

//----------------------------------------------------------------------------//

/// Returns the chromosomes with non-dominated fitness found so far.
template <class Decoder>
std::vector<Chromosome> NSBRKGA<Decoder>::getIncumbentChromosomes() const {
    std::vector<Chromosome> result;

    for (std::size_t i = 0; i < this->incumbent_solutions.size(); i++) {
        result.push_back(this->incumbent_solutions[i].second);
    }

    return result;
}

//----------------------------------------------------------------------------//

/// Returns the solutions with non-dominated fitness found so far.
template <class Decoder>
const std::vector<std::pair<std::vector<double>, Chromosome>> &
NSBRKGA<Decoder>::getIncumbentSolutions() const {
    return this->incumbent_solutions;
}

//----------------------------------------------------------------------------//

/**
 * \brief Returns the fitness of a chromosome of the given population.
 * \param population_index the population index.
 * \param position the chromosome position, ordered by fitness.
 *        The best chromosome is located in position 0.
 * \throw std::range_error either if `population_index` is larger
 *        than number of populations, or `position` is larger than the
 *        population size.
 */
template <class Decoder>
std::vector<double> NSBRKGA<Decoder>::getFitness(unsigned population_index,
                                                 unsigned position) const {
    if (population_index >= this->current.size()) {
        throw std::range_error("The population index is larger than number of "
                               "populations");
    }

    if (position >= this->params.population_size) {
        throw std::range_error("The chromosome position is larger than number "
                               "of populations");
    }

    return this->current[population_index]->fitness[position].first;
}

//----------------------------------------------------------------------------//

/**
 * \brief Returns a reference to a chromosome of the given population.
 * \param population_index the population index.
 * \param position the chromosome position, ordered by fitness.
 *        The best chromosome is located in position 0.
 * \throw std::range_error either if `population_index` is larger
 *        than number of populations, or `position` is larger than the
 *        population size.
 */
template <class Decoder>
const Chromosome &NSBRKGA<Decoder>::getChromosome(unsigned population_index,
                                                  unsigned position) const {
    if (population_index >= this->current.size()) {
        throw std::range_error("The population index is larger than number of "
                               "populations");
    }

    if (position >= this->params.population_size) {
        throw std::range_error("The chromosome position is larger than number "
                               "of populations");
    }

    return this->current[population_index]->getChromosome(position);
}

//----------------------------------------------------------------------------//

/**
 * \brief Injects a chromosome and its fitness into a population in the
 *         given place position.
 *
 * If fitness is not provided (`fitness == Inf`), the decoding is performed
 * over chromosome. Once the chromosome is injected, the population is
 * re-sorted according to the chromosomes' fitness.
 *
 * \param chromosome the chromosome to be injected.
 * \param population_index the population index.
 * \param position the chromosome position.
 *
 * \throw std::range_error either if `population_index` is larger
 *        than number of populations; or `position` is larger than the
 *        population size; or ` chromosome.size() != chromosome_size`
 */
template <class Decoder>
void NSBRKGA<Decoder>::injectChromosome(const Chromosome &chromosome,
                                        unsigned population_index,
                                        unsigned position) {
    if (population_index >= this->current.size()) {
        throw std::range_error("The population index is larger than number of "
                               "populations");
    }

    if (position >= this->params.population_size) {
        throw std::range_error("The chromosome position is larger than number "
                               "of populations");
    }

    if (chromosome.size() != this->CHROMOSOME_SIZE) {
        throw std::range_error("Wrong chromosome size");
    }

    auto &pop = this->current[population_index];
    auto &local_chr = pop->population[pop->fitness[position].second];
    local_chr = chromosome;
    std::vector<double> fitness = this->decoder.decode(local_chr, true);

    pop->setFitness(position, fitness);
    pop->sortFitness(this->OPT_SENSES);

    this->updateIncumbentSolutions(
        std::vector<std::pair<std::vector<double>, Chromosome>>(
            1, std::make_pair(fitness, chromosome)));
}

//----------------------------------------------------------------------------//

/**
 * \brief Sets a custom bias function used to build the probabilities.
 *
 * It must be a **positive non-increasing function**, i.e.
 * \f$ f: \mathbb{N}^+ \to \mathbb{R}^+\f$ such that
 * \f$f(i) \ge 0\f$ and \f$f(i) \ge f(i+1)\f$ for
 * \f$i \in [1..total\_parents]\f$.
 * For example
 * \code{.cpp}
 *      setBiasCustomFunction(
 *          [](const unsigned x) {
 *              return 1.0 / (x * x);
 *          }
 *      );
 * \endcode
 * sets an inverse quadratic function.
 *
 * \param func a reference to a unary positive non-increasing function.
 * \throw std::runtime_error in case the function is not a non-decreasing
 *        positive function.
 */
template <class Decoder>
void NSBRKGA<Decoder>::setBiasCustomFunction(
    const std::function<double(const unsigned)> &func) {

    std::vector<double> bias_values(this->params.total_parents);
    std::iota(bias_values.begin(), bias_values.end(), 1);
    std::transform(bias_values.begin(), bias_values.end(), bias_values.begin(),
                   func);

    // If it is not non-increasing, throw an error.
    if (!std::is_sorted(bias_values.rbegin(), bias_values.rend())) {
        throw std::runtime_error("bias_function must be positive "
                                 "non-decreasing");
    }

    if (this->bias_function) {
        this->params.bias_type = BiasFunctionType::CUSTOM;
    }

    this->bias_function = func;
    this->total_bias_weight =
        std::accumulate(bias_values.begin(), bias_values.end(), 0.0);
}

//----------------------------------------------------------------------------//

/**
 * \brief Sets a custom diversity function used to build the elite set.
 */
template <class Decoder>
void NSBRKGA<Decoder>::setDiversityCustomFunction(
    const std::function<double(const std::vector<std::vector<double>> &)>
        &func) {
    this->diversity_function = func;
}

//----------------------------------------------------------------------------//

/**
 * \brief Resets all populations with brand new keys.
 *
 * All warm-start solutions provided setInitialPopulation() are discarded.
 * You may use injectChromosome() to insert those solutions again.
 * \param intensity the intensity of the reset.
 * \param population_index the index of the population to be reset. If
 * `population_index >= num_independent_populations`, all populations
 * are reset.
 * \throw std::runtime_error if the algorithm is not initialized.
 */
template <class Decoder>
void NSBRKGA<Decoder>::reset(double intensity, unsigned population_index) {
    if (!this->initialized) {
        throw std::runtime_error("The algorithm hasn't been initialized. "
                                 "Don't forget to call initialize() method");
    }

    unsigned pop_start = population_index;
    unsigned pop_end = population_index;

    if (population_index >= this->params.num_independent_populations) {
        pop_start = 0;
        pop_end = this->params.num_independent_populations - 1;
    }

    for (; pop_start <= pop_end; pop_start++) {
        for (unsigned i = 0; i < this->current[pop_start]->getPopulationSize();
             i++) {
            if (this->rand01() < intensity) {
                for (unsigned j = 0; j < this->CHROMOSOME_SIZE; j++) {
                    (*this->current[pop_start])(i, j) = this->rand01();
                }
            }
        }

        std::vector<std::pair<std::vector<double>, Chromosome>> new_solutions(
            this->params.population_size);

#ifdef _OPENMP
#pragma omp parallel for num_threads(MAX_THREADS) schedule(static, 1)
#endif
        for (unsigned j = 0; j < this->params.population_size; j++) {
            this->current[pop_start]->setFitness(
                j, this->decoder.decode((*this->current[pop_start])(j), true));
            new_solutions[j] =
                std::make_pair(this->current[pop_start]->getFitness(j),
                               (*this->current[pop_start])(j));
        }

        this->updateIncumbentSolutions(new_solutions);

        // Now we must sort by fitness, since things might have changed.
        this->current[pop_start]->sortFitness(this->OPT_SENSES);
    }
}

//----------------------------------------------------------------------------//

/**
 * \brief Evolves the current populations following the guidelines of
 *        NSBRKGAs.
 *
 * \warning
 *     The decoding is done in parallel using threads, and the user **must
 *     guarantee that the decoder is THREAD-SAFE.** If such property cannot
 *     be held, we suggest using a single thread for optimization.
 *
 * \param generations number of generations to be evolved. Must be larger
 *        than zero.
 * \throw std::runtime_error if the algorithm is not initialized.
 * \throw std::range_error if the number of generations is zero.
 */
template <class Decoder> bool NSBRKGA<Decoder>::evolve(unsigned generations) {
    if (!this->initialized) {
        throw std::runtime_error("The algorithm hasn't been initialized. "
                                 "Don't forget to call initialize() method");
    }

    if (generations == 0) {
        throw std::range_error("Cannot evolve for 0 generations.");
    }

    bool result = false;

    for (unsigned i = 0; i < generations; i++) {
        for (unsigned j = 0; j < this->params.num_independent_populations;
             j++) {
            // First evolve the population (current, next).
            if (this->evolution(*(this->current)[j], *(this->previous)[j])) {
                result = true;
            }

            std::swap(this->current[j], this->previous[j]);
        }
    }

    return result;
}

//----------------------------------------------------------------------------//

/**
 * \brief Exchanges elite-solutions between the populations.
 *
 * Given a population, the `num_immigrants` best solutions are copied to
 * the neighbor populations, replacing their worth solutions. If there is
 * only one population, nothing is done.
 *
 * \param num_immigrants number of elite chromosomes to select from each
 *      population.
 * \throw std::range_error if the number of immigrants less than one or
 *        it is larger than or equal to the population size divided by
 *        the number of populations minus one, i.e. \f$\lceil
 *        \frac{population\_size}{num\_independent\_populations} \rceil
 *         - 1\f$.
 */
template <class Decoder>
void NSBRKGA<Decoder>::exchangeElite(unsigned num_immigrants) {
    if (this->params.num_independent_populations == 1) {
        return;
    }

    unsigned immigrants_threshold =
        ceil(this->params.population_size /
             (this->params.num_independent_populations - 1));

    if (num_immigrants < 1 || num_immigrants >= immigrants_threshold) {
        std::stringstream ss;
        ss << "Number of immigrants (" << num_immigrants
           << ") less than one, "
              "or larger than or equal to population size / "
              "num_independent_populations ("
           << immigrants_threshold << ")";
        throw std::range_error(ss.str());
    }

#ifdef _OPENMP
#pragma omp parallel for num_threads(MAX_THREADS)
#endif
    for (unsigned i = 0; i < this->params.num_independent_populations; i++) {
        // Population i will receive some elite members from each Population j.
        // Last chromosome of i (will be overwritten below).
        unsigned dest = this->params.population_size - 1;
        for (unsigned j = 0; j < this->params.num_independent_populations;
             j++) {
            if (j == i) {
                continue;
            }

            // Copy the num_immigrants best from Population j into Population i.
            for (unsigned m = 0; m < num_immigrants; m++) {
                // Copy the m-th best of Population j into the 'dest'-th
                // position of Population i
                const auto best_of_j = this->current[j]->getChromosome(m);
                std::copy(best_of_j.begin(), best_of_j.end(),
                          this->current[i]->getChromosome(dest).begin());
                this->current[i]->fitness[dest].first =
                    this->current[j]->fitness[m].first;
                dest--;
            }
        }
    }

// Re-sort each population since they were modified.
#ifdef _OPENMP
#pragma omp parallel for num_threads(MAX_THREADS)
#endif
    for (unsigned i = 0; i < this->params.num_independent_populations; i++) {
        this->current[i]->sortFitness(this->OPT_SENSES);
    }
}

//----------------------------------------------------------------------------//

/**
 * \brief Sets individuals to initial population.
 *
 * Set initial individuals into the population to work as warm-starters.
 * Such individuals can be obtained from solutions of external procedures
 * such as fast heuristics, other metaheuristics, or even relaxations from
 * a mixed integer programming model that models the problem.
 *
 * All given solutions are assigned to one population only. Therefore, the
 * maximum number of solutions is the size of the populations.
 *
 * \param populations a set of individuals encoded as Chromosomes.
 * \throw std::runtime_error if the number of given chromosomes is larger
 *        than the population size; if the sizes of the given chromosomes
 *        do not match with the required chromosome size.
 */
template <class Decoder>
void NSBRKGA<Decoder>::setInitialPopulations(
    const std::vector<std::vector<Chromosome>> &populations) {
    if (populations.size() > this->params.num_independent_populations) {
        std::stringstream ss;
        ss << "Number of given populations (" << populations.size() << ") is "
           << "larger than the maximum number of independent populations ("
           << this->params.num_independent_populations << ")";
        throw std::runtime_error(ss.str());
    }

    for (std::size_t i = 0; i < populations.size(); i++) {
        std::vector<Chromosome> chromosomes = populations[i];

        if (chromosomes.size() > this->params.population_size) {
            std::stringstream ss;
            ss << "Error on setting initial population " << i << ": number of "
               << "given chromosomes (" << chromosomes.size() << ") is larger "
               << "than the population size (" << this->params.population_size
               << ")";
            throw std::runtime_error(ss.str());
        }

        this->current[i].reset(new Population(
            this->CHROMOSOME_SIZE, chromosomes.size(), this->diversity_function,
            this->min_num_elites, this->max_num_elites));

        for (std::size_t j = 0; j < chromosomes.size(); j++) {
            Chromosome chr = chromosomes[j];

            if (chr.size() != this->CHROMOSOME_SIZE) {
                std::stringstream ss;
                ss << "Error on setting initial population " << i << ": "
                   << "chromosome " << j << " does not have the required "
                   << "dimension (actual size: " << chr.size() << ", required "
                   << "size: " << this->CHROMOSOME_SIZE << ")";
                throw std::runtime_error(ss.str());
            }

            std::copy(chr.begin(), chr.end(),
                      this->current[i]->population[j].begin());
        }
    }

    this->initial_populations = true;
}

//----------------------------------------------------------------------------//

/**
 * \brief Initializes the populations and others parameters of the
 *        algorithm.
 *
 * If a initial population is supplied, this method completes the remain
 * individuals, if they do not exist. This method also performs the initial
 * decoding of the chromosomes. Therefore, depending on the decoder
 * implementation, this can take a while, and the user may want to time
 * such procedure in his/her experiments.
 *
 * \warning
 *      This method must be call before any evolutionary or population
 *      handling method.
 *
 * \warning
 *     As it is in #evolve(), the decoding is done in parallel using
 *     threads, and the user **must guarantee that the decoder is
 *     THREAD-SAFE.** If such property cannot be held, we suggest using
 *     a single thread  for optimization.
 */
template <class Decoder> void NSBRKGA<Decoder>::initialize() {
    // Verify the initial population and complete or prune it!
    if (this->initial_populations) {
        for (unsigned i = 0; i < this->params.num_independent_populations;
             i++) {
            auto pop = this->current[i];

            if (pop->population.size() < this->params.population_size) {
                Chromosome chromosome(this->CHROMOSOME_SIZE);
                std::size_t j = pop->population.size();

                pop->population.resize(this->params.population_size);
                pop->fitness.resize(this->params.population_size);

                for (; j < this->params.population_size; j++) {
                    for (unsigned k = 0; k < this->CHROMOSOME_SIZE; k++) {
                        chromosome[k] = this->rand01();
                    }

                    pop->population[j] = chromosome;
                }
            }
            // Prune some additional chromosomes.
            else if (pop->population.size() > this->params.population_size) {
                pop->population.resize(this->params.population_size);
                pop->fitness.resize(this->params.population_size);
            }
        }
    } else {
        // Initialize each chromosome of the current population.
        for (unsigned i = 0; i < this->params.num_independent_populations;
             i++) {
            this->current[i].reset(new Population(
                this->CHROMOSOME_SIZE, this->params.population_size,
                this->diversity_function, this->min_num_elites,
                this->max_num_elites));

            for (unsigned j = 0; j < this->params.population_size; j++) {
                for (unsigned k = 0; k < this->CHROMOSOME_SIZE; k++) {
                    (*this->current[i])(j, k) = this->rand01();
                }
            }
        }
    }

    std::vector<std::pair<std::vector<double>, Chromosome>> new_solutions(
        this->params.num_independent_populations *
        this->params.population_size);

    // Initialize and decode each chromosome of the current population,
    // then copy to previous.
    for (unsigned i = 0; i < this->params.num_independent_populations; i++) {
#ifdef _OPENMP
#pragma omp parallel for num_threads(MAX_THREADS) schedule(static, 1)
#endif
        for (unsigned j = 0; j < this->params.population_size; j++) {
            this->current[i]->setFitness(
                j, this->decoder.decode((*this->current[i])(j), true));
            new_solutions[i * this->params.population_size + j] =
                std::make_pair(this->current[i]->getFitness(j),
                               (*this->current[i])(j));
        }

        // Sort and copy to previous.
        this->current[i]->sortFitness(this->OPT_SENSES);

        this->previous[i].reset(new Population(*this->current[i]));
    }

    this->updateIncumbentSolutions(new_solutions);

    this->initialized = true;
}

//----------------------------------------------------------------------------//

/**
 * \brief Performs a shaking in the chosen population.
 * \param intensity the intensity of the shaking.
 * \param distribution the distribution of the shaking.
 * \param population_index the index of the population to be shaken. If
 * `population_index >= num_independent_populations`, all populations
 * are shaken.
 * \throw std::runtime_error if the algorithm is not initialized.
 */
template <class Decoder>
void NSBRKGA<Decoder>::shake(double intensity, double distribution,
                             unsigned population_index) {
    if (!this->initialized) {
        throw std::runtime_error("The algorithm hasn't been initialized. "
                                 "Don't forget to call initialize() method");
    }

    unsigned pop_start = population_index;
    unsigned pop_end = population_index;

    if (population_index >= this->params.num_independent_populations) {
        pop_start = 0;
        pop_end = this->params.num_independent_populations - 1;
    }

    for (; pop_start <= pop_end; pop_start++) {
        for (unsigned i = 0; i < this->current[pop_start]->getPopulationSize();
             i++) {
            for (unsigned j = 0; j < this->CHROMOSOME_SIZE; j++) {
                this->polynomialMutation((*this->current[pop_start])(i, j),
                                         intensity, distribution);
            }
        }

        std::vector<std::pair<std::vector<double>, Chromosome>> new_solutions(
            this->params.population_size);

#ifdef _OPENMP
#pragma omp parallel for num_threads(MAX_THREADS) schedule(static, 1)
#endif
        for (unsigned j = 0; j < this->params.population_size; j++) {
            this->current[pop_start]->setFitness(
                j, this->decoder.decode((*this->current[pop_start])(j), true));
            new_solutions[j] =
                std::make_pair(this->current[pop_start]->getFitness(j),
                               (*this->current[pop_start])(j));
        }

        this->updateIncumbentSolutions(new_solutions);

        // Now we must sort by fitness, since things might have changed.
        this->current[pop_start]->sortFitness(this->OPT_SENSES);
    }
}

//---------------------------------------------------------------------------//

template <class Decoder>
void NSBRKGA<Decoder>::selectParents(const Population &curr, const size_t &chr,
                                     const bool use_best_individual) {
    // Rebuild the indices.
    std::iota(this->shuffled_individuals.begin(),
              this->shuffled_individuals.end(), 0);

    if (use_best_individual) {
        // Take one of the best individuals.
        this->parents_indexes[0] = chr - curr.num_elites;
    }

    // Shuffles the elite set.
    std::shuffle(this->shuffled_individuals.begin(),
                 this->shuffled_individuals.begin() + curr.num_elites,
                 this->rng);

    // Take the elite parents indexes.
    if (!use_best_individual) {
        for (unsigned j = 0; j < this->params.num_elite_parents; j++) {
            this->parents_indexes[j] = this->shuffled_individuals[j];
        }
    } else {
        for (unsigned j = 1; j < this->params.num_elite_parents; j++) {
            this->parents_indexes[j] = this->shuffled_individuals[j - 1];
        }
    }

    // Shuffles the whole population.
    std::shuffle(this->shuffled_individuals.begin(),
                 this->shuffled_individuals.end(), this->rng);

    // Take the remaining parents indexes.
    for (unsigned j = this->params.num_elite_parents;
         j < this->params.total_parents; j++) {
        this->parents_indexes[j] =
            this->shuffled_individuals[j - this->params.num_elite_parents];
    }

    // Sorts the parents indexes
    std::sort(this->parents_indexes.begin(), this->parents_indexes.end());

    for (unsigned j = 0; j < this->params.total_parents; j++) {
        this->parents_ordered[j] = curr.fitness[this->parents_indexes[j]];
    }
}

//---------------------------------------------------------------------------//

/**
 * @brief Applies polynomial mutation to a single allele value.
 *
 * Performs polynomial mutation on the given allele, a variation operator
 * commonly used in multi-objective evolutionary algorithms (e.g., NSGA-II).
 * The mutation perturbs the allele value using a polynomial probability
 * distribution, where the shape of the distribution is controlled by the
 * mutation distribution index.
 *
 * The mutation is applied probabilistically based on the given mutation
 * probability. When triggered, a random perturbation (delta) is computed
 * using the polynomial distribution, which favors small perturbations
 * near the original value while still allowing larger changes.
 *
 * The resulting allele value is clamped to the valid range [0.0, 1.0),
 * where the upper bound is set to RAND_MAX / (RAND_MAX + 1.0) to ensure
 * the allele remains strictly less than 1.0.
 *
 * @param[in,out] allele The allele value to be mutated, expected to be
 *                       in the range [0.0, 1.0). Modified in place if
 *                       mutation is applied.
 * @param[in] mutation_probability The probability of applying the mutation
 *                                 to the allele (value in [0.0, 1.0]).
 * @param[in] mutation_distribution The distribution index (eta_m) that
 *                                  controls the shape of the polynomial
 *                                  distribution. Higher values produce
 *                                  perturbations closer to the original
 *                                  value. Note: this parameter is unused
 *                                  in the body; the member variable
 *                                  `this->params.mutation_distribution`
 *                                  is used instead.
 */
template <class Decoder>
void NSBRKGA<Decoder>::polynomialMutation(double &allele,
                                          double mutation_probability,
                                          double mutation_distribution) {
    if (this->rand01() < mutation_probability) {
        double y = allele,
               inner_exponent = (this->params.mutation_distribution + 1.0),
               outer_exponent =
                   1.0 / (this->params.mutation_distribution + 1.0),
               delta_l = y - 0.0, delta_r = 1.0 - y, delta = 0.0,
               u = this->rand01();

        if (u < 0.5) {
            delta =
                std::pow(2.0 * u + (1.0 - 2.0 * u) *
                                       std::pow(1.0 - delta_l, inner_exponent),
                         outer_exponent) -
                1.0;
        } else {
            delta =
                1.0 - std::pow(2.0 * (1.0 - u) +
                                   2.0 * (u - 0.5) *
                                       std::pow(1.0 - delta_r, inner_exponent),
                               outer_exponent);
        }

        allele += delta;

        if (allele < 0.0) {
            allele = 0.0;
        } else if (allele >= 1.0) {
            allele = ((double)RAND_MAX) / ((double)RAND_MAX + 1.0);
        }
    }
}

//---------------------------------------------------------------------------//

template <class Decoder>
void NSBRKGA<Decoder>::polynomialMutation(double &allele,
                                          double mutation_probability) {
    this->polynomialMutation(allele, mutation_probability,
                             this->params.mutation_distribution);
}

//---------------------------------------------------------------------------//

template <class Decoder>
void NSBRKGA<Decoder>::polynomialMutation(double &allele) {
    this->polynomialMutation(allele, this->params.mutation_probability,
                             this->params.mutation_distribution);
}

//---------------------------------------------------------------------------//

template <class Decoder>
void NSBRKGA<Decoder>::mate(const Population &curr, Chromosome &offspring) {
    const unsigned P = this->params.total_parents;

    // Precompute bias weights using global population rank of each parent.
    this->total_bias_weight = 0.0;
    for (const unsigned &i : this->parents_indexes) {
        this->total_bias_weight += this->bias_function(i + 1);
    }

    switch (this->params.crossover_type) {
    //--------------------------------------------------------------
    // GEOMETRIC: biased weighted-average crossover.
    // For each parent j (0-based), r = parents_indexes[j] + 1 is
    // its 1-based rank in the population.  The random weight is
    // drawn from Uniform(phi(r), phi(r+1)).
    //--------------------------------------------------------------
    case CrossoverType::GEOMETRIC: {
        for (unsigned gene = 0; gene < this->CHROMOSOME_SIZE; ++gene) {
            double weighted_sum = 0.0;
            double weight_total = 0.0;

            for (unsigned j = 0; j < P; ++j) {
                const unsigned r = this->parents_indexes[j] + 1;
                const double a = this->bias_function(r);
                const double b = this->bias_function(r + 1);
                const double lo = std::min(a, b);
                const double hi = std::max(a, b);
                // w_r ~ Uniform(lo, hi)
                const double w = lo + this->rand01() * (hi - lo);

                weighted_sum += w * curr(this->parents_ordered[j].second, gene);
                weight_total += w;
            }

            offspring[gene] = weighted_sum / weight_total;

            // Performs the polynomial mutation.
            this->polynomialMutation(offspring[gene]);
        }
        break;
    }

    //--------------------------------------------------------------
    // ROULETTE (default): discrete biased roulette crossover.
    //--------------------------------------------------------------
    case CrossoverType::ROULETTE:
    default: {
        for (unsigned gene = 0; gene < this->CHROMOSOME_SIZE; ++gene) {
            // Roulette method using global population rank.
            unsigned parent = 0;
            double cumulative_probability = 0.0;
            const double toss = this->rand01();

            do {
                // Start parent from 1 because the bias function.
                cumulative_probability +=
                    this->bias_function(this->parents_indexes[parent++] + 1) /
                    this->total_bias_weight;
            } while (cumulative_probability < toss);

            // Decrement parent to the right index, and take the allele.
            offspring[gene] =
                curr(this->parents_ordered[--parent].second, gene);

            // Performs the polynomial mutation.
            this->polynomialMutation(offspring[gene]);
        }
        break;
    }
    } // switch crossover_type
}

//---------------------------------------------------------------------------//

/**
 * \brief Evolves the current population to the next.
 *
 * Note that the next population will be re-populate completely.
 *
 * \param[in] curr current population.
 * \param[out] next next population.
 */
template <class Decoder>
bool NSBRKGA<Decoder>::evolution(Population &curr, Population &next) {
    bool result = false;
    Chromosome offspring(this->CHROMOSOME_SIZE);

    // First, we copy the elite chromosomes to the next generation.
    for (unsigned chr = 0; chr < curr.num_elites; chr++) {
        next.population[chr] = curr.population[curr.fitness[chr].second];
        next.fitness[chr] = std::make_pair(curr.fitness[chr].first, chr);
    }

    // Second, we generate 'num_objectives' offspring,
    // always using one of the best individuals.
    for (std::size_t chr = curr.num_elites;
         chr < curr.num_elites + this->OPT_SENSES.size(); chr++) {
        // Selects the parents.
        this->selectParents(curr, chr, true);

        // Performs the mate.
        this->mate(curr, offspring);

        // This strategy of setting the offpring in a local variable,
        // and then copying to the population seems to reduce the
        // overall cache misses counting.
        next.getChromosome(chr) = offspring;
    }

    // Third, we generate 'pop_size - num_elites - num_objectives' offspring.
    for (std::size_t chr = curr.num_elites + this->OPT_SENSES.size();
         chr < this->params.population_size; chr++) {
        // Selects the parents.
        this->selectParents(curr, chr);

        // Performs the mate.
        this->mate(curr, offspring);

        // This strategy of setting the offpring in a local variable,
        // and then copying to the population seems to reduce the
        // overall cache misses counting.
        next.getChromosome(chr) = offspring;
    }

    std::vector<std::pair<std::vector<double>, Chromosome>> new_solutions(
        this->params.population_size - curr.num_elites);

// Time to compute fitness, in parallel.
#ifdef _OPENMP
#pragma omp parallel for num_threads(MAX_THREADS) schedule(static, 1)
#endif
    for (unsigned i = curr.num_elites; i < this->params.population_size; i++) {
        next.setFitness(i, this->decoder.decode(next.population[i], true));
        new_solutions[i - curr.num_elites] =
            std::make_pair(next.fitness[i].first, next.population[i]);
    }

    if (this->updateIncumbentSolutions(new_solutions)) {
        result = true;
    }

    // Now we must sort by fitness, since things might have changed.
    next.sortFitness(this->OPT_SENSES);

    return result;
}

//----------------------------------------------------------------------------//

/**
 * \brief Performs path relinking between elite solutions that are, at least,
 * a given minimum distance between themselves. In this method, the
 * local/loaded parameters are ignored in favor to the supplied ones.
 *
 * In the presence of multiple populations, the path relinking is performed
 * between elite chromosomes from different populations, in a circular
 * fashion. For example, suppose we have 3 populations. The framework
 * performs 3 path relinkings: the first between individuals from
 * populations 1 and 2, the second between populations 2 and 3, and the
 * third between populations 3 and 1. In the case of just one population,
 * both base and guiding individuals are sampled from the elite set of that
 * population.
 *
 * Note that the algorithm tries to find a pair of base and guiding
 * solutions with a minimum distance given by the distance function. If this
 * is not possible, a new pair of solutions are sampled (without
 * replacement) and tested against the distance. In case it is not possible
 * to find such pairs for the given populations, the algorithm skips to the
 * next pair of populations (in a circular fashion, as described above).
 * Yet, if such pairs are not found in any case, the algorithm declares
 * failure. This indicates that the populations are very homogeneous.
 *
 * If the found solution is the best solution found so far, IPR replaces the
 * worst solution by it. Otherwise, IPR computes the distance between the
 * found solution and all other solutions in the elite set, and replaces the
 * worst solution by it if and only if the found solution is, at least,
 * `minimum_distance` from all them.
 *
 * The API will call `Decoder::decode()` function always with `rewrite =
 * false`. The reason is that if the decoder rewrites the chromosome, the
 * path between solutions is lost and inadvertent results may come up. Note
 * that at the end of the path relinking, the method calls the decoder with
 * `rewrite = true` in the best chromosome found to guarantee that this
 * chromosome is re-written to reflect the best solution found.
 *
 * This method is a multi-thread implementation. Instead of to build and
 * decode each chromosome one at a time, the method builds a list of
 * candidates, altering the alleles/keys according to the guide solution,
 * and then decode all candidates in parallel. Note that
 * `O(chromosome_size^2 / block_size)` additional memory is necessary to
 * build the candidates, which can be costly if the `chromosome_size` is
 * very large.
 *
 * \warning
 *     As it is in #evolve(), the decoding is done in parallel using
 *     threads, and the user **must guarantee that the decoder is
 *     THREAD-SAFE.** If such property cannot be held, we suggest using
 *     a single thread  for optimization.
 *
 * \param pr_type type of path relinking to be performed.
 *        See PathRelinking::Type.
 * \param dist a pointer to a functor/object to compute the distance between
 *        two chromosomes. This object must be inherited from
 *        NSBRKGA::DistanceFunctionBase and implement its methods.
 * \param max_time aborts path relinking when reach `max_time`.
 *        If `max_time <= 0`, no limit is imposed.
 *        Default: 0 (no limit).
 * \param percentage defines the size, in percentage, of the path to build.
 *        Default: 1.0 (100%).
 *
 * \returns A PathRelinking::PathRelinkingResult depending on the relink
 *          status.
 *
 * \throw std::range_error if the percentage or size of the path is
 *        not in (0, 1].
 */
template <class Decoder>
PathRelinking::PathRelinkingResult
NSBRKGA<Decoder>::pathRelink(PathRelinking::Type pr_type,
                             std::shared_ptr<DistanceFunctionBase> dist,
                             long max_time, double percentage) {

    using PR = PathRelinking::PathRelinkingResult;

    if (max_time <= 0) {
        max_time = std::numeric_limits<long>::max();
    }

    double max_distance = 0.0;
    std::pair<std::vector<double>, Chromosome> initial_solution,
        guiding_solution, best_solution;
    std::vector<std::pair<std::vector<double>, Chromosome>> best_solutions;

    initial_solution.first.resize(this->OPT_SENSES.size());
    initial_solution.second.resize(this->CHROMOSOME_SIZE);
    guiding_solution.first.resize(this->OPT_SENSES.size());
    guiding_solution.second.resize(this->CHROMOSOME_SIZE);

    // Keep track of the time.
    this->pr_start_time = std::chrono::system_clock::now();

    auto final_status = PR::NO_IMPROVEMENT;

    // Perform path relinking between elite chromosomes from different
    // populations. This is done in a circular fashion.
    for (unsigned pop_count = 0;
         pop_count < this->params.num_independent_populations; pop_count++) {
        auto elapsed_seconds =
            std::chrono::duration_cast<std::chrono::seconds>(
                std::chrono::system_clock::now() - this->pr_start_time)
                .count();
        if (elapsed_seconds > max_time) {
            break;
        }

        unsigned pop_base = pop_count;
        unsigned pop_guide = pop_count + 1;

        // If we have just one population, we take the both solution from it.
        if (this->params.num_independent_populations == 1) {
            pop_base = pop_guide = 0;
            pop_count = this->params.num_independent_populations;
        }
        // If we have two populations, perform just one path relinking.
        else if (this->params.num_independent_populations == 2) {
            pop_count = this->params.num_independent_populations;
        }

        // Do the circular thing.
        if (pop_guide == this->params.num_independent_populations) {
            pop_guide = 0;
        }

        // Find the pair of elite chromosomes with the largest distance.
        max_distance = 0;

        for (std::size_t i = 0; i < this->current[pop_base]->num_elites; i++) {
            for (std::size_t j = 0; j < this->current[pop_guide]->num_elites;
                 j++) {
                const auto &fit1 = this->current[pop_base]->fitness[0].first;
                const auto &chr1 =
                    this->current[pop_base]->population
                        [this->current[pop_base]->fitness[0].second];

                const auto &fit2 = this->current[pop_guide]->fitness[0].first;
                const auto &chr2 =
                    this->current[pop_guide]->population
                        [this->current[pop_guide]->fitness[0].second];

                const double distance = dist->distance(chr1, chr2);

                if (max_distance < distance) {
                    copy(begin(fit1), end(fit1), begin(initial_solution.first));
                    copy(begin(chr1), end(chr1),
                         begin(initial_solution.second));
                    copy(begin(fit2), end(fit2), begin(guiding_solution.first));
                    copy(begin(chr2), end(chr2),
                         begin(guiding_solution.second));
                    max_distance = distance;
                }
            }
        }

        best_solutions.clear();

        // Perform the path relinking.
        if (pr_type == PathRelinking::Type::ALLOCATION) {
            best_solution = this->allocationPathRelink(
                initial_solution, guiding_solution, max_time, percentage,
                best_solutions);
        } else if (pr_type == PathRelinking::Type::PERMUTATION) {
            best_solution = this->permutationPathRelink(
                initial_solution, guiding_solution, max_time, percentage,
                best_solutions);
        } else { // pr_type == PathRelinking::Type::BINARY_SEARCH
            best_solution = this->binarySearchPathRelink(
                initial_solution, guiding_solution, max_time, best_solutions);
        }

        // Re-decode and apply local search if the decoder are able to do it.
        best_solution.first = this->decoder.decode(best_solution.second, true);

        NSBRKGA<Decoder>::updateIncumbentSolutions(
            best_solutions, {best_solution}, this->OPT_SENSES,
            this->params.num_incumbent_solutions);

        if (this->dominates(best_solution.first, initial_solution.first) &&
            this->dominates(best_solution.first, guiding_solution.first)) {
            final_status |= PR::ELITE_IMPROVEMENT;
        }

        // Include the best solution found in the population.
        std::copy(begin(best_solution.second), end(best_solution.second),
                  begin(this->current[pop_base]->population
                            [this->current[pop_base]->fitness.back().second]));
        this->current[pop_base]->fitness.back().first = best_solution.first;
        // Reorder the chromosomes.
        this->current[pop_base]->sortFitness(this->OPT_SENSES);

        if (this->updateIncumbentSolutions(best_solutions)) {
            final_status |= PR::BEST_IMPROVEMENT;
        }
    }

    return final_status;
}

//----------------------------------------------------------------------------//

/**
 * \brief Performs path relinking between elite solutions that are,
 *        at least, a given minimum distance between themselves.
 *
 * This method uses all parameters supplied in the constructor.
 * In particular, the block size is computed by
 * \f$\lceil \alpha \times \sqrt{p} \rceil\f$
 * where \f$\alpha\f$ is NsbrkgaParams#alpha_block_size and
 * \f$p\f$ is NsbrkgaParams#population_size.
 * If the size is larger than the chromosome size, the size is set to
 * half of the chromosome size.
 *
 * Please, refer to #pathRelink() for details.
 *
 * \param dist a pointer to a functor/object to compute the distance between
 *        two chromosomes. This object must be inherited from
 *        NSBRKGA::DistanceFunctionBase and implement its methods.
 * \param max_time aborts path relinking when reach `max_time`.
 *        If `max_time <= 0`, no limit is imposed.
 *        Default: 0 (no limit).
 * \param percentage defines the size, in percentage, of the path to build.
 *        Default: 1.0 (100%).
 *
 * \returns A PathRelinking::PathRelinkingResult depending on the relink
 *          status.
 *
 * \throw std::range_error if the percentage or size of the path is
 *        not in (0, 1].
 */
template <class Decoder>
PathRelinking::PathRelinkingResult
NSBRKGA<Decoder>::pathRelink(std::shared_ptr<DistanceFunctionBase> dist,
                             long max_time, double percentage) {

    return this->pathRelink(this->params.pr_type, dist, max_time, percentage);
}

//----------------------------------------------------------------------------//

// This is a multi-thread version. For small chromosomes, it may be slower than
// single thread version.
/**
 * \brief Performs the direct path relinking.
 *
 * This method changes each allele or block of alleles of base chromosome
 * for the correspondent one in the guide chromosome.
 *
 * This method is a multi-thread implementation. Instead of to build and
 * decode each chromosome one at a time, the method builds a list of
 * candidates, altering the alleles/keys according to the guide solution,
 * and then decode all candidates in parallel. Note that
 * `O(chromosome_size^2)` additional memory is necessary to
 * build the candidates, which can be costly if the `chromosome_size` is
 * very large.
 *
 * \param solution1 first solution (fitness–chromosome pair).
 * \param solution2 second solution (fitness–chromosome pair).
 * \param max_time abort path relinking when reach `max_time`.
 *        If `max_time <= 0`, no limit is imposed.
 * \param percentage define the size, in percentage, of the path to build.
 * \param[out] best_solutions the best solutions found in the search.
 * \return the best solution found in the search.
 */
template <class Decoder>
std::pair<std::vector<double>, Chromosome>
NSBRKGA<Decoder>::allocationPathRelink(
    const std::pair<std::vector<double>, Chromosome> &solution1,
    const std::pair<std::vector<double>, Chromosome> &solution2, long max_time,
    double percentage,
    std::vector<std::pair<std::vector<double>, Chromosome>> &best_solutions) {
    const std::size_t PATH_SIZE =
        std::size_t(percentage * this->CHROMOSOME_SIZE);
    // Create a empty solution.
    std::pair<std::vector<double>, Chromosome> best_solution;
    best_solution.second.resize(this->CHROMOSOME_SIZE, 0.0);

    best_solution.first = std::vector<double>(this->OPT_SENSES.size());
    for (std::size_t m = 0; m < this->OPT_SENSES.size(); m++) {
        if (this->OPT_SENSES[m] == Sense::MAXIMIZE) {
            best_solution.first[m] = std::numeric_limits<double>::lowest();
        } else {
            best_solution.first[m] = std::numeric_limits<double>::max();
        }
    }

    // Create the vector of indices to test.
    std::vector<std::size_t> remaining_genes(this->CHROMOSOME_SIZE);
    std::iota(remaining_genes.begin(), remaining_genes.end(), 0);
    std::shuffle(remaining_genes.begin(), remaining_genes.end(), this->rng);

    Chromosome old_keys(this->CHROMOSOME_SIZE);

    struct Triple {
      public:
        Chromosome chr;
        std::vector<double> fitness;
        std::vector<std::size_t>::iterator it_gene_index;
        Triple() : chr(), fitness(0), it_gene_index() {}
    };

    // Allocate memory for the candidates.
    std::vector<Triple> candidates_left(this->CHROMOSOME_SIZE);
    std::vector<Triple> candidates_right(this->CHROMOSOME_SIZE);

    for (std::size_t i = 0; i < candidates_left.size(); i++) {
        candidates_left[i].chr.resize(this->CHROMOSOME_SIZE);
    }

    for (std::size_t i = 0; i < candidates_right.size(); i++) {
        candidates_right[i].chr.resize(this->CHROMOSOME_SIZE);
    }

    Chromosome chr1(solution1.second);
    Chromosome chr2(solution2.second);

    Chromosome *base = &chr1;
    Chromosome *guide = &chr2;
    std::vector<Triple> *candidates_base = &candidates_left;
    std::vector<Triple> *candidates_guide = &candidates_right;

#ifdef _OPENMP
#pragma omp parallel for num_threads(MAX_THREADS)
#endif
    for (std::size_t i = 0; i < candidates_left.size(); i++) {
        std::copy(begin(*base), end(*base), begin(candidates_left[i].chr));
    }

#ifdef _OPENMP
#pragma omp parallel for num_threads(MAX_THREADS)
#endif
    for (std::size_t i = 0; i < candidates_right.size(); i++) {
        std::copy(begin(*guide), end(*guide), begin(candidates_right[i].chr));
    }

    std::vector<std::pair<std::vector<double>, Chromosome>> candidate_solutions;

    std::size_t iterations = 0;
    while (!remaining_genes.empty()) {
        // Set the keys from the guide solution for each candidate.
        std::vector<std::size_t>::iterator it_gene_idx =
            remaining_genes.begin();
        for (std::size_t i = 0; i < remaining_genes.size(); i++) {
            // Save the former keys before...
            std::copy_n((*candidates_base)[i].chr.begin() + (*it_gene_idx), 1,
                        old_keys.begin() + (*it_gene_idx));

            // ... copy the keys from the guide solution.
            std::copy_n(guide->begin() + (*it_gene_idx), 1,
                        (*candidates_base)[i].chr.begin() + (*it_gene_idx));

            (*candidates_base)[i].it_gene_index = it_gene_idx;
            it_gene_idx++;
        }

        // Decode the candidates.
        volatile bool times_up = false;
#ifdef _OPENMP
#pragma omp parallel for num_threads(MAX_THREADS) shared(times_up)             \
    schedule(static, 1)
#endif
        for (std::size_t i = 0; i < remaining_genes.size(); i++) {
            (*candidates_base)[i].fitness =
                std::vector<double>(this->OPT_SENSES.size());

            for (std::size_t m = 0; m < this->OPT_SENSES.size(); m++) {
                if (this->OPT_SENSES[m] == Sense::MAXIMIZE) {
                    (*candidates_base)[i].fitness[m] =
                        std::numeric_limits<double>::lowest();
                } else {
                    (*candidates_base)[i].fitness[m] =
                        std::numeric_limits<double>::max();
                }
            }

            if (times_up) {
                continue;
            }

            (*candidates_base)[i].fitness =
                this->decoder.decode((*candidates_base)[i].chr, false);

            const auto elapsed_seconds =
                std::chrono::duration_cast<std::chrono::seconds>(
                    std::chrono::system_clock::now() - this->pr_start_time)
                    .count();

            if (elapsed_seconds > max_time) {
                times_up = true;
            }
        }

        candidate_solutions.resize(remaining_genes.size());
        std::transform((*candidates_base).begin(),
                       (*candidates_base).begin() + remaining_genes.size(),
                       candidate_solutions.begin(), [](Triple candidate) {
                           return std::make_pair(candidate.fitness,
                                                 candidate.chr);
                       });
        NSBRKGA<Decoder>::updateIncumbentSolutions(
            best_solutions, candidate_solutions, this->OPT_SENSES,
            this->params.num_incumbent_solutions);

        // Locate the best candidate.
        std::size_t best_index = 0;
        std::vector<std::size_t>::iterator best_it_gene_index;

        std::vector<double> best_value(this->OPT_SENSES.size());
        for (std::size_t m = 0; m < this->OPT_SENSES.size(); m++) {
            if (this->OPT_SENSES[m] == Sense::MAXIMIZE) {
                best_value[m] = std::numeric_limits<double>::lowest();
            } else {
                best_value[m] = std::numeric_limits<double>::max();
            }
        }

        for (std::size_t i = 0; i < remaining_genes.size(); i++) {
            if (this->dominates((*candidates_base)[i].fitness, best_value)) {
                best_it_gene_index = (*candidates_base)[i].it_gene_index;
                best_value = (*candidates_base)[i].fitness;
                best_index = i;
            }
        }

        // Hold it, if it is the best found until now.
        if (this->dominates((*candidates_base)[best_index].fitness,
                            best_solution.first)) {
            best_solution.first = (*candidates_base)[best_index].fitness;
            std::copy(begin((*candidates_base)[best_index].chr),
                      end((*candidates_base)[best_index].chr),
                      begin(best_solution.second));
        }

        // Restore original keys and copy the keys for all future candidates.
        // The last candidate will not be used.
        it_gene_idx = remaining_genes.begin();
        for (std::size_t i = 0; i < remaining_genes.size() - 1;
             i++, it_gene_idx++) {
            if (i != best_index) {
                std::copy_n(old_keys.begin() + (*it_gene_idx), 1,
                            (*candidates_base)[i].chr.begin() + (*it_gene_idx));
            }
            std::copy_n(
                (*candidates_base)[best_index].chr.begin() +
                    (*best_it_gene_index),
                1, (*candidates_base)[i].chr.begin() + (*best_it_gene_index));
        }

        std::copy_n((*candidates_base)[best_index].chr.begin() +
                        (*best_it_gene_index),
                    1, base->begin() + (*best_it_gene_index));

        std::swap(base, guide);
        std::swap(candidates_base, candidates_guide);
        remaining_genes.erase(best_it_gene_index);

        const auto elapsed_seconds =
            std::chrono::duration_cast<std::chrono::seconds>(
                std::chrono::system_clock::now() - this->pr_start_time)
                .count();

        if ((elapsed_seconds > max_time) || (iterations++ > PATH_SIZE)) {
            break;
        }
    } // end while

    return best_solution;
}

//----------------------------------------------------------------------------//

/**
 * \brief Performs the permutation-based path relinking.
 *
 * In this method, the permutation induced by the keys in the guide
 * solution is used to change the order of the keys in the permutation
 * induced by the base solution.
 *
 * This method is a multi-thread implementation. Instead of to build and
 * decode each chromosome one at a time, the method builds a list of
 * candidates, altering the alleles/keys according to the guide solution,
 * and then decode all candidates in parallel. Note that
 * `O(chromosome_size^2)` additional memory is necessary to
 * build the candidates, which can be costly if the `chromosome_size` is
 * very large.
 *
 * The path relinking is performed by changing the order of
 * each allele of base chromosome for the correspondent one in
 * the guide chromosome.
 * \param solution1 first solution (fitness–chromosome pair).
 * \param solution2 second solution (fitness–chromosome pair).
 * \param max_time abort path relinking when reach `max_time`.
 *        If `max_time <= 0`, no limit is imposed.
 * \param percentage define the size, in percentage, of the path to build.
 * \param[out] best_solutions the best solutions found in the search.
 * \return the best solution found in the search.
 */
template <class Decoder>
std::pair<std::vector<double>, Chromosome>
NSBRKGA<Decoder>::permutationPathRelink(
    const std::pair<std::vector<double>, Chromosome> &solution1,
    const std::pair<std::vector<double>, Chromosome> &solution2, long max_time,
    double percentage,
    std::vector<std::pair<std::vector<double>, Chromosome>> &best_solutions) {
    const std::size_t PATH_SIZE =
        std::size_t(percentage * this->CHROMOSOME_SIZE);
    // Create a empty solution.
    std::pair<std::vector<double>, Chromosome> best_solution;
    best_solution.second.resize(this->CHROMOSOME_SIZE, 0.0);

    best_solution.first = std::vector<double>(this->OPT_SENSES.size());
    for (std::size_t m = 0; m < this->OPT_SENSES.size(); m++) {
        if (this->OPT_SENSES[m] == Sense::MAXIMIZE) {
            best_solution.first[m] = std::numeric_limits<double>::lowest();
        } else {
            best_solution.first[m] = std::numeric_limits<double>::max();
        }
    }

    std::vector<std::size_t> remaining_indices(this->CHROMOSOME_SIZE);
    std::iota(remaining_indices.begin(), remaining_indices.end(), 0);
    std::shuffle(remaining_indices.begin(), remaining_indices.end(), this->rng);

    struct DecodeStruct {
      public:
        Chromosome chr;
        std::vector<double> fitness;
        std::vector<std::size_t>::iterator key_index_it;
        std::size_t pos1;
        std::size_t pos2;
        DecodeStruct() : chr(), fitness(0), key_index_it(0), pos1(0), pos2(0) {}
    };

    // Allocate memory for the candidates.
    std::vector<DecodeStruct> candidates_left(this->CHROMOSOME_SIZE);
    std::vector<DecodeStruct> candidates_right(this->CHROMOSOME_SIZE);

    for (std::size_t i = 0; i < candidates_left.size(); i++) {
        candidates_left[i].chr.resize(this->CHROMOSOME_SIZE);
    }

    for (std::size_t i = 0; i < candidates_right.size(); i++) {
        candidates_right[i].chr.resize(this->CHROMOSOME_SIZE);
    }

    Chromosome chr1(solution1.second);
    Chromosome chr2(solution2.second);

    Chromosome *base = &chr1;
    Chromosome *guide = &chr2;
    std::vector<DecodeStruct> *candidates_base = &candidates_left;
    std::vector<DecodeStruct> *candidates_guide = &candidates_right;

    std::vector<std::size_t> chr1_indices(this->CHROMOSOME_SIZE);
    std::vector<std::size_t> chr2_indices(this->CHROMOSOME_SIZE);
    std::vector<std::size_t> *base_indices = &chr1_indices;
    std::vector<std::size_t> *guide_indices = &chr2_indices;

    // Create and order the indices.
    std::vector<std::pair<std::vector<double>, std::size_t>> sorted(
        this->CHROMOSOME_SIZE);

    for (unsigned j = 0; j < 2; j++) {
        for (std::size_t i = 0; i < base->size(); i++) {
            sorted[i] =
                std::pair<std::vector<double>, std::size_t>((*base)[i], i);
        }

        std::sort(begin(sorted), end(sorted));
        for (std::size_t i = 0; i < base->size(); i++) {
            (*base_indices)[i] = sorted[i].second;
        }

        swap(base, guide);
        swap(base_indices, guide_indices);
    }

    base = &chr1;
    guide = &chr2;
    base_indices = &chr1_indices;
    guide_indices = &chr2_indices;

#ifdef _OPENMP
#pragma omp parallel for num_threads(MAX_THREADS)
#endif
    for (std::size_t i = 0; i < candidates_left.size(); i++) {
        std::copy(begin(*base), end(*base), begin(candidates_left[i].chr));
    }

#ifdef _OPENMP
#pragma omp parallel for num_threads(MAX_THREADS)
#endif
    for (std::size_t i = 0; i < candidates_right.size(); i++) {
        std::copy(begin(*guide), end(*guide), begin(candidates_right[i].chr));
    }

    std::vector<std::pair<std::vector<double>, Chromosome>> candidate_solutions;

    std::size_t iterations = 0;
    while (!remaining_indices.empty()) {
        std::size_t position_in_base;
        std::size_t position_in_guide;

        std::vector<std::size_t>::iterator it_idx = remaining_indices.begin();
        for (std::size_t i = 0; i < remaining_indices.size(); i++) {
            position_in_base = (*base_indices)[*it_idx];
            position_in_guide = (*guide_indices)[*it_idx];

            if (position_in_base == position_in_guide) {
                it_idx = remaining_indices.erase(it_idx);
                i--;
                continue;
            }

            (*candidates_base)[i].key_index_it = it_idx;
            (*candidates_base)[i].pos1 = position_in_base;
            (*candidates_base)[i].pos2 = position_in_guide;
            (*candidates_base)[i].fitness =
                std::vector<double>(this->OPT_SENSES.size());
            for (unsigned m = 0; m < this->OPT_SENSES.size(); m++) {
                if (this->OPT_SENSES[m] == Sense::MAXIMIZE) {
                    (*candidates_base)[i].fitness[m] =
                        std::numeric_limits<double>::lowest();
                } else {
                    (*candidates_base)[i].fitness[m] =
                        std::numeric_limits<double>::max();
                }
            }

            it_idx++;
        }

        if (remaining_indices.size() == 0) {
            break;
        }

        // Decode the candidates.
        volatile bool times_up = false;
#ifdef _OPENMP
#pragma omp parallel for num_threads(MAX_THREADS) shared(times_up)             \
    schedule(static, 1)
#endif
        for (std::size_t i = 0; i < remaining_indices.size(); i++) {
            if (times_up) {
                continue;
            }

            std::swap((*candidates_base)[i].chr[(*candidates_base)[i].pos1],
                      (*candidates_base)[i].chr[(*candidates_base)[i].pos2]);

            (*candidates_base)[i].fitness =
                this->decoder.decode((*candidates_base)[i].chr, false);

            std::swap((*candidates_base)[i].chr[(*candidates_base)[i].pos1],
                      (*candidates_base)[i].chr[(*candidates_base)[i].pos2]);

            const auto elapsed_seconds =
                std::chrono::duration_cast<std::chrono::seconds>(
                    std::chrono::system_clock::now() - this->pr_start_time)
                    .count();
            if (elapsed_seconds > max_time) {
                times_up = true;
            }
        }

        candidate_solutions.resize(remaining_indices.size());
        std::transform((*candidates_base).begin(),
                       (*candidates_base).begin() + remaining_indices.size(),
                       candidate_solutions.begin(), [](DecodeStruct candidate) {
                           return std::make_pair(candidate.fitness,
                                                 candidate.chr);
                       });
        NSBRKGA<Decoder>::updateIncumbentSolutions(
            best_solutions, candidate_solutions, this->OPT_SENSES,
            this->params.num_incumbent_solutions);

        // Locate the best candidate
        std::vector<std::size_t>::iterator best_key_index_it;

        std::size_t best_index;
        std::vector<double> best_value(this->OPT_SENSES.size());
        for (unsigned m = 0; m < this->OPT_SENSES.size(); m++) {
            if (this->OPT_SENSES[m] == Sense::MAXIMIZE) {
                best_value[m] = std::numeric_limits<double>::lowest();
            } else {
                best_value[m] = std::numeric_limits<double>::max();
            }
        }

        for (std::size_t i = 0; i < remaining_indices.size(); i++) {
            if (this->dominates((*candidates_base)[i].fitness, best_value)) {
                best_index = i;
                best_key_index_it = (*candidates_base)[i].key_index_it;
                best_value = (*candidates_base)[i].fitness;
            }
        }

        position_in_base = (*base_indices)[*best_key_index_it];
        position_in_guide = (*guide_indices)[*best_key_index_it];

        // Commit the best exchange in all candidates.
        // The last will not be used.
        for (std::size_t i = 0; i < remaining_indices.size() - 1; i++) {
            std::swap((*candidates_base)[i].chr[position_in_base],
                      (*candidates_base)[i].chr[position_in_guide]);
        }

        std::swap((*base_indices)[position_in_base],
                  (*base_indices)[position_in_guide]);

        // Hold, if it is the best found until now
        if (this->dominates(best_value, best_solution.first)) {
            const auto &best_chr = (*candidates_base)[best_index].chr;
            best_solution.first = best_value;
            copy(begin(best_chr), end(best_chr), begin(best_solution.second));
        }

        std::swap(base_indices, guide_indices);
        std::swap(candidates_base, candidates_guide);
        remaining_indices.erase(best_key_index_it);

        // Is time to stop?
        const auto elapsed_seconds =
            std::chrono::duration_cast<std::chrono::seconds>(
                std::chrono::system_clock::now() - this->pr_start_time)
                .count();

        if ((elapsed_seconds > max_time) || (iterations++ > PATH_SIZE)) {
            break;
        }
    }

    return best_solution;
}

//----------------------------------------------------------------------------//

/**
 * \brief Performs the binary-search-based path relinking.
 *
 * \param solution1 first solution (fitness–chromosome pair).
 * \param solution2 second solution (fitness–chromosome pair).
 * \param max_time abort path relinking when reach `max_time`.
 *        If `max_time <= 0`, no limit is imposed.
 * \param[out] best_solutions the best solutions found in the search.
 * \return the best solution found in the search.
 */
template <class Decoder>
std::pair<std::vector<double>, Chromosome>
NSBRKGA<Decoder>::binarySearchPathRelink(
    const std::pair<std::vector<double>, Chromosome> &solution1,
    const std::pair<std::vector<double>, Chromosome> &solution2, long max_time,
    std::vector<std::pair<std::vector<double>, Chromosome>> &best_solutions) {
    std::pair<std::vector<double>, Chromosome> best_solution, mid_solution,
        left_solution, right_solution;
    long elapsed_seconds = 0;
    double lambda;

    best_solution.second.resize(this->CHROMOSOME_SIZE, 0.0);
    best_solution.first.resize(this->OPT_SENSES.size());
    for (std::size_t m = 0; m < this->OPT_SENSES.size(); m++) {
        if (this->OPT_SENSES[m] == Sense::MAXIMIZE) {
            best_solution.first[m] = std::numeric_limits<double>::lowest();
        } else {
            best_solution.first[m] = std::numeric_limits<double>::max();
        }
    }
    mid_solution.second.resize(this->CHROMOSOME_SIZE);
    mid_solution.first.resize(this->OPT_SENSES.size());
    left_solution.second = Chromosome(solution1.second);
    left_solution.first = std::vector<double>(solution1.first);
    right_solution.second = Chromosome(solution2.second);
    right_solution.first = std::vector<double>(solution2.first);

    while (elapsed_seconds < max_time &&
           !std::equal(
               left_solution.first.begin(), left_solution.first.end(),
               right_solution.first.begin(),
               [](double a, double b) {
                   return fabs(a - b) < std::numeric_limits<double>::epsilon();
               }) &&
           !std::equal(
               left_solution.second.begin(), left_solution.second.end(),
               right_solution.second.begin(),
               [](double a, double b) {
                   return fabs(a - b) < std::numeric_limits<double>::epsilon();
               })) {
        lambda = this->rand01();
        std::transform(left_solution.second.begin(), left_solution.second.end(),
                       right_solution.second.begin(),
                       mid_solution.second.begin(),
                       [lambda](double a, double b) {
                           return lambda * a + (1.0 - lambda) * b;
                       });
        mid_solution.first = this->decoder.decode(mid_solution.second, false);

        if (this->dominates(mid_solution.first, best_solution.first)) {
            best_solution.second = Chromosome(mid_solution.second);
            best_solution.first = std::vector<double>(mid_solution.first);
        }

        NSBRKGA<Decoder>::updateIncumbentSolutions(
            best_solutions, {mid_solution}, this->OPT_SENSES,
            this->params.num_incumbent_solutions);

        if (this->dominates(left_solution.first, right_solution.first)) {
            right_solution.second = Chromosome(mid_solution.second);
            right_solution.first = std::vector<double>(mid_solution.first);
        } else if (this->dominates(right_solution.first, left_solution.first)) {
            left_solution.second = Chromosome(mid_solution.second);
            left_solution.first = std::vector<double>(mid_solution.first);
        } else if (lambda < 0.5) {
            left_solution.second = Chromosome(mid_solution.second);
            left_solution.first = std::vector<double>(mid_solution.first);
        } else {
            right_solution.second = Chromosome(mid_solution.second);
            right_solution.first = std::vector<double>(mid_solution.first);
        }

        elapsed_seconds =
            std::chrono::duration_cast<std::chrono::seconds>(
                std::chrono::system_clock::now() - this->pr_start_time)
                .count();
    }

    return best_solution;
}

//----------------------------------------------------------------------------//

template <class Decoder>
bool NSBRKGA<Decoder>::updateIncumbentSolutions(
    std::vector<std::pair<std::vector<double>, Chromosome>>
        &incumbent_solutions,
    const std::vector<std::pair<std::vector<double>, Chromosome>>
        &new_solutions,
    const std::vector<Sense> &senses, const std::size_t max_num_solutions) {
    bool result = false;

    if (new_solutions.empty()) {
        return result;
    }

    std::vector<std::pair<std::vector<double>, Chromosome>> sorted_solutions =
        Population::nonDominatedSort<Chromosome>(new_solutions, senses).front();

    for (const std::pair<std::vector<double>, Chromosome> &new_solution :
         sorted_solutions) {
        bool is_dominated_or_equal = false;

        for (std::vector<std::pair<std::vector<double>, Chromosome>>::iterator
                 it = incumbent_solutions.begin();
             it != incumbent_solutions.end();) {
            const std::pair<std::vector<double>, Chromosome>
                &incumbent_solution = *it;

            if (Population::dominates(new_solution.first,
                                      incumbent_solution.first, senses)) {
                it = incumbent_solutions.erase(it);
            } else {
                if (Population::dominates(incumbent_solution.first,
                                          new_solution.first, senses) ||
                    std::equal(
                        incumbent_solution.first.begin(),
                        incumbent_solution.first.end(),
                        new_solution.first.begin(), [](double a, double b) {
                            return fabs(a - b) <
                                   std::numeric_limits<double>::epsilon();
                        })) {
                    is_dominated_or_equal = true;
                    break;
                }

                it++;
            }
        }

        if (!is_dominated_or_equal) {
            incumbent_solutions.push_back(new_solution);
            result = true;
        }
    }

    if (max_num_solutions > 0 &&
        incumbent_solutions.size() > max_num_solutions) {
        Population::crowdingSort<Chromosome>(incumbent_solutions);
        incumbent_solutions.resize(max_num_solutions);
        result = true;
    }

    return result;
}

//----------------------------------------------------------------------------//

/**
 * @brief Updates the incumbent (best known) solutions with new candidate
 * solutions.
 *
 * This is a convenience member function that delegates to the static version of
 * updateIncumbentSolutions, passing the instance's incumbent solutions,
 * optimization senses, and the configured maximum number of incumbent
 * solutions.
 *
 * @tparam Decoder The decoder class used to evaluate chromosomes.
 *
 * @param new_solutions A vector of pairs, where each pair contains an objective
 *        values vector and its corresponding chromosome, representing the new
 *        candidate solutions to be considered for inclusion in the incumbent
 * set.
 *
 * @return True if the incumbent solutions were updated (i.e., at least one new
 *         solution was added or replaced an existing one); false otherwise.
 */
template <class Decoder>
bool NSBRKGA<Decoder>::updateIncumbentSolutions(
    const std::vector<std::pair<std::vector<double>, Chromosome>>
        &new_solutions) {
    return NSBRKGA<Decoder>::updateIncumbentSolutions(
        this->incumbent_solutions, new_solutions, this->OPT_SENSES,
        this->params.num_incumbent_solutions);
}

//----------------------------------------------------------------------------//

/**
 * @brief Generates a random double-precision floating-point number
 *        uniformly distributed in the range [0, 1).
 *
 * Uses std::generate_canonical to produce a high-precision random value
 * with as many random bits as the double type supports. This method is
 * preferred over manual scaling of the RNG output to avoid precision
 * issues that can occur on certain platforms (e.g., Linux).
 *
 * @return A uniformly distributed random double in the interval [0, 1).
 */
template <class Decoder> inline double NSBRKGA<Decoder>::rand01() {
    // **NOTE:** instead to use std::generate_canonical<> (which can be
    // a little bit slow), we may use
    //    rng() * (1.0 / std::numeric_limits<std::mt19937::result_type>::max());
    // However, this approach has some precision problems on some platforms
    // (notably Linux)

    return std::generate_canonical<double, std::numeric_limits<double>::digits>(
        this->rng);
}

//----------------------------------------------------------------------------//

/**
 * @brief Generates a uniformly distributed random integer in the range [0, n].
 *
 * Uses a rejection sampling technique adapted from Magnus Jonsson
 * (magnus@smartelectronix.com) to produce an unbiased random integer.
 * The method works by first computing a bitmask that covers all bits
 * used by the value @p n, then repeatedly drawing random values masked
 * to those bits until the result falls within the desired range [0, n].
 * This avoids the modulo bias that would result from a simple
 * modulus operation.
 *
 * @tparam Decoder The decoder type used by the NSBRKGA framework.
 * @param n The upper bound (inclusive) of the random integer range.
 *          Must be a non-negative value representable by uint_fast32_t.
 * @return A uniformly distributed random integer in the range [0, n].
 *
 * @note This method is specific to uint_fast32_t types (up to 32-bit values).
 * @note The expected number of iterations is at most 2, since the mask
 *       ensures at least half of the drawn values fall within [0, n].
 */
template <class Decoder>
inline uint_fast32_t NSBRKGA<Decoder>::randInt(const uint_fast32_t n) {
    // This code was adapted from Magnus Jonsson (magnus@smartelectronix.com)
    // Find which bits are used in n. Note that this is specific
    // for uint_fast32_t types.

    uint_fast32_t used = n;
    used |= used >> 1;
    used |= used >> 2;
    used |= used >> 4;
    used |= used >> 8;
    used |= used >> 16;

    // Draw numbers until one is found in [0, n].
    uint_fast32_t i;
    do {
        i = this->rng() & used; // Toss unused bits to shorten search.
    } while (i > n);
    return i;
}

} // end namespace NSBRKGA

//----------------------------------------------------------------------------//
// Template specializations for enum I/O
//----------------------------------------------------------------------------//

/**
 * \defgroup template_specs Template specializations for enum I/O.
 *
 * Using slightly modified template class provided by Bradley Plohr
 * (https://codereview.stackexchange.com/questions/14309/conversion-between-enum-and-string-in-c-class-header)
 * we specialize that template to enums in the BRKGA namespace.
 *
 * The EnumIO class helps to read and write enums from streams directly,
 * saving time in coding custom solutions. Please, see third_part/enum_io.hpp
 * for complete reference and examples.
 *
 * \note
 *      The specializations must be done in the global namespace.
 *
 * \warning The specialization must be inline-d to avoid multiple definitions
 * issues across different modules. However, this can cause "inline" overflow,
 * and compromise your code. If you include this header only once along with
 * your code, it is safe to remove the `inline`s from the specializations. But,
 * if this is not the case, you should move these specializations to a module
 * you know is included only once, for instance, the `main()` module.
 */
///@{

/// \cond

/// Template specialization to NSBRKGA::Sense.
template <>
INLINE const std::vector<std::string> &EnumIO<NSBRKGA::Sense>::enum_names() {
    static std::vector<std::string> enum_names_({"MINIMIZE", "MAXIMIZE"});
    return enum_names_;
}

/// Template specialization to NSBRKGA::PathRelinking::Type.
template <>
INLINE const std::vector<std::string> &
EnumIO<NSBRKGA::PathRelinking::Type>::enum_names() {
    static std::vector<std::string> enum_names_(
        {"ALLOCATION", "PERMUTATION", "BINARY_SEARCH"});
    return enum_names_;
}

/// Template specialization to NSBRKGA::BiasFunctionType.
template <>
INLINE const std::vector<std::string> &
EnumIO<NSBRKGA::BiasFunctionType>::enum_names() {
    static std::vector<std::string> enum_names_(
        {"CONSTANT", "CUBIC", "EXPONENTIAL", "LINEAR", "LOGINVERSE",
         "QUADRATIC", "SQRT", "CBRT", "CUSTOM"});
    return enum_names_;
}

/// Template specialization to NSBRKGA::DiversityFunctionType.
template <>
INLINE const std::vector<std::string> &
EnumIO<NSBRKGA::DiversityFunctionType>::enum_names() {
    static std::vector<std::string> enum_names_(
        {"NONE", "AVERAGE_DISTANCE_TO_CENTROID",
         "AVERAGE_DISTANCE_BETWEEN_ALL_PAIRS", "POWER_MEAN_BASED", "CUSTOM"});
    return enum_names_;
}

/// Template specialization to NSBRKGA::DistanceFunctionType.
template <>
INLINE const std::vector<std::string> &
EnumIO<NSBRKGA::DistanceFunctionType>::enum_names() {
    static std::vector<std::string> enum_names_(
        {"HAMMING", "KENDALL_TAU", "EUCLIDEAN", "CUSTOM"});
    return enum_names_;
}

/// Template specialization to NSBRKGA::CrossoverType.
template <>
INLINE const std::vector<std::string> &
EnumIO<NSBRKGA::CrossoverType>::enum_names() {
    static std::vector<std::string> enum_names_({"ROULETTE", "GEOMETRIC"});
    return enum_names_;
}

/// \endcond
///@}

#endif // NSBRKGA_HPP_
