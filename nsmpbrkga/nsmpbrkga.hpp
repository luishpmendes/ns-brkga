/******************************************************************************
 * nsmpbrkga.hpp: Non-dominated Sorting Multi-Parent Biased Random-Key Genetic
 *                Algorithm Multi-Parent.
 *
 * (c) Copyright 2015-2020, Carlos Eduardo de Andrade.
 * All Rights Reserved.
 *
 * (c) Copyright 2010, 2011 Rodrigo F. Toso, Mauricio G.C. Resende.
 * All Rights Reserved.
 *
 * Created on : Jan 06, 2015 by andrade.
 * Last update: Aug 12, 2020 by luishpmendes.
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

#ifndef NSMPBRKGA_HPP_
#define NSMPBRKGA_HPP_

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

/// If we need to include this file in multiple translation units (files) that
/// are compiled separately, we have to `inline` some functions and template
/// definitions to avoid multiple definitions and linking problems. However,
/// such inlining can make the object code grows large. In other cases, the
/// compiler may complain about too many inline functions, if you are already
/// using several inline functions.
#ifdef NSMPBRKGA_MULTIPLE_INCLUSIONS
    #define INLINE inline
#else
    #define INLINE
#endif

/**
 * \brief This namespace contains all stuff related to NSMPBRKGA.
 */
namespace BRKGA {

//----------------------------------------------------------------------------//
// Enumerations
//----------------------------------------------------------------------------//

/// Specifies objective as minimization or maximization.
enum class Sense {
    MINIMIZE = false,  ///< Minimization.
    MAXIMIZE = true    ///< Maximization.
};

/// Holds the enumerations for Path Relinking algorithms.
namespace PathRelinking {

/// Specifies type of path relinking.
enum class Type {
    /// Changes each key for the correspondent in the other chromosome.
    DIRECT,

    /// Switches the order of a key for that in the other chromosome.
    PERMUTATION
};

/// Specifies which individuals used to build the path.
enum class Selection {
    /// Selects, in the order, the best solution of each population.
    BESTSOLUTION,

    /// Chooses uniformly random solutions from the elite sets.
    RANDOMELITE
};

/// Specifies the result type/status of path relink procedure.
enum class PathRelinkingResult {
    /// The chromosomes among the populations are too homogeneous and the path
    /// relink will not generate improved solutions.
    TOO_HOMOGENEOUS = 0,

    /// Path relink was done but no improved solution was found.
    NO_IMPROVEMENT = 1,

    /// An improved solution among the elite set was found, but the best
    /// solution was not improved.
    ELITE_IMPROVEMENT = 3,

    /// The best solution was improved.
    BEST_IMPROVEMENT = 7
};

/**
 *  \brief Perform bitwise `OR` between two `PathRelinkingResult` returning
 *         the highest rank `PathRelinkingResult`.
 *
 *  For example
 *  - TOO_HOMOGENEOUS | NO_IMPROVEMENT == NO_IMPROVEMENT
 *  - NO_IMPROVEMENT | ELITE_IMPROVEMENT == ELITE_IMPROVEMENT
 *  - ELITE_IMPROVEMENT | BEST_IMPROVEMENT == BEST_IMPROVEMENT
 */
inline PathRelinkingResult & operator|=(PathRelinkingResult & lhs,
                                        PathRelinkingResult rhs) {
    lhs = PathRelinkingResult(static_cast<unsigned>(lhs) |
                              static_cast<unsigned>(rhs));
    return lhs;
}
} // namespace PathRelinking

/// Specifies a bias function type when choosing parents to mating
/// (`r` is a given parameter). This function substitutes the `rho`
/// parameter from the original BRKGA.
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

    /// Indicates a custom function supplied by the user.
    CUSTOM
};

/// Specifies the type of shaking to be performed.
enum class ShakingType {
    /// Applies the following perturbations:
    /// 1. Inverts the value of a random chosen, i.e., from `value` to
    ///    `1 - value`;
    /// 2. Assigns a random value to a random key.
    CHANGE = 0,

    /// Applies two swap perturbations:
    /// 1. Swaps the values of a randomly chosen key `i` and its
    ///    neighbor `i + 1`;
    /// 2. Swaps values of two randomly chosen keys.
    SWAP = 1
};

//----------------------------------------------------------------------------//
// Distance functions
//----------------------------------------------------------------------------//

/**
 * \brief Distance Function Base.
 *
 * This class is a interface for functors that compute the distance between
 * two vectors of double numbers.
 */
class DistanceFunctionBase {
public:
    /// Default constructor.
    DistanceFunctionBase() {}

    /// Default destructor.
    virtual ~DistanceFunctionBase() {}

    /**
     * \brief Computes the distance between two vectors.
     * \param v1 first vector
     * \param v2 second vector
     */
    virtual double distance(const std::vector<double> & v1,
                            const std::vector<double> & v2) = 0;

    /**
     * \brief Returns true if the changing of `key1` by `key2` affects
     *        the solution.
     * \param key1 the first key
     * \param key2 the second key
     */
    virtual bool affectSolution(const double key1, const double key2) = 0;

    /**
     * \brief Returns true if the changing of the blocks of keys `v1` by the
     *        blocks of keys `v2` affects the solution.
     * \param v1_begin begin of the first blocks of keys
     * \param v2_begin begin of the first blocks of keys
     * \param block_size number of keys to be considered.
     */
    virtual bool affectSolution(
            std::vector<double>::const_iterator v1_begin,
            std::vector<double>::const_iterator v2_begin,
            const std::size_t block_size) = 0;
};

//----------------------------------------------------------------------------//

/**
 * \brief Hamming distance between two vectors.
 *
 * This class is a functor that computes the Hamming distance between two
 * vectors. It takes a threshold parameter to "binarize" the vectors.
 */
class HammingDistance: public DistanceFunctionBase {
public:
    /**
     * \brief Default constructor
     * \param _threshold used to rounding the values to 0 or 1.
     */
    explicit HammingDistance(const double _threshold = 0.5):
        threshold(_threshold) {}

    /// Default destructor
    virtual ~HammingDistance() {}

    /**
     * \brief Computes the Hamming distance between two vectors.
     * \param vector1 first vector
     * \param vector2 second vector
     */
    virtual double distance(const std::vector<double> & vector1,
                            const std::vector<double> & vector2) {
        if(vector1.size() != vector2.size()) {
            throw std::runtime_error("The size of the vector must "
                                     "be the same!");
        }

        int dist = 0;
        for(std::size_t i = 0; i < vector1.size(); ++i) {
            if((vector1[i] < threshold) != (vector2[i] < this->threshold)) {
                ++dist;
            }
        }

        return dist;
    }

    /**
     * \brief Returns true if the changing of `key1` by `key2` affects
     *        the solution.
     * \param key1 the first key
     * \param key2 the second key
     */
    virtual bool affectSolution(const double key1, const double key2) {
        return (key1 < this->threshold ? 0 : 1)
            != (key2 < this->threshold ? 0 : 1);
    }

    /**
     * \brief Returns true if the changing of the blocks of keys `v1` by the
     *        blocks of keys `v2` affects the solution.
     * \param v1_begin begin of the first blocks of keys
     * \param v2_begin begin of the first blocks of keys
     * \param block_size number of keys to be considered.
     */
    virtual bool affectSolution(
            std::vector<double>::const_iterator v1_begin,
            std::vector<double>::const_iterator v2_begin,
            const std::size_t block_size) {
        for(std::size_t i = 0; i < block_size;
            ++i, ++v1_begin, ++v2_begin) {
            if((*v1_begin < this->threshold) 
            != (*v2_begin < this->threshold)) {
                return true;
            }
        }
        return false;
    }

public:
    /// Threshold parameter used to rounding the values to 0 or 1.
    double threshold;
};

//----------------------------------------------------------------------------//

/**
 * \brief Kendall Tau distance between two vectors.
 *
 * This class is a functor that computes the Kendall Tau distance between two
 * vectors. This version is not normalized.
 */
class KendallTauDistance: public DistanceFunctionBase {
public:
    /// Default constructor.
    KendallTauDistance() {}

    /// Default destructor.
    virtual ~KendallTauDistance() {}

    /**
     * \brief Computes the Kendall Tau distance between two vectors.
     * \param vector1 first vector
     * \param vector2 second vector
     */
    virtual double distance(const std::vector<double> & vector1,
                            const std::vector<double> & vector2) {
        if(vector1.size() != vector2.size()) {
            throw std::runtime_error("The size of the vector must "
                                     "be the same!");
        }

        const std::size_t size = vector1.size();

        std::vector<std::pair<double, std::size_t>> pairs_v1;
        std::vector<std::pair<double, std::size_t>> pairs_v2;

        pairs_v1.reserve(size);
        std::size_t rank = 0;
        for(unsigned i = 0; i < vector1.size(); i++) {
            pairs_v1.emplace_back(vector1[i], ++rank);
        }

        pairs_v2.reserve(size);
        rank = 0;
        for(unsigned i = 0; i < vector2.size(); i++) {
            pairs_v2.emplace_back(vector2[i], ++rank);
        }

        std::sort(begin(pairs_v1), end(pairs_v1));
        std::sort(begin(pairs_v2), end(pairs_v2));

        unsigned disagreements = 0;
        for(std::size_t i = 0; i < size - 1; ++i) {
            for(std::size_t j = i + 1; j < size; ++j) {
                if((pairs_v1[i].second < pairs_v1[j].second
                    && pairs_v2[i].second > pairs_v2[j].second) ||
                   (pairs_v1[i].second > pairs_v1[j].second
                    && pairs_v2[i].second < pairs_v2[j].second)) {
                    ++disagreements;
                }
            }
        }

        return double(disagreements);
    }

    /**
     * \brief Returns true if the changing of `key1` by `key2` affects
     *        the solution.
     * \param key1 the first key
     * \param key2 the second key
     */
    virtual bool affectSolution(const double key1, const double key2) {
        return fabs(key1 - key2) > 1e-6;
    }

    /**
     * \brief Returns true if the changing of the blocks of keys `v1` by the
     *        blocks of keys `v2` affects the solution.
     *
     * \param v1_begin begin of the first blocks of keys
     * \param v2_begin begin of the first blocks of keys
     * \param block_size number of keys to be considered.
     *
     * \todo (ceandrade): implement this properly.
     */
    virtual bool affectSolution(
            std::vector<double>::const_iterator v1_begin,
            std::vector<double>::const_iterator v2_begin,
            const std::size_t block_size) {
        return block_size == 1?
              affectSolution(*v1_begin, *v2_begin) : true;
    }
};

//----------------------------------------------------------------------------//
// Population class.
//----------------------------------------------------------------------------//

/**
 * \brief Encapsulates a population of chromosomes.
 *
 * Encapsulates a population of chromosomes providing supporting methods for
 * making the implementation easier.
 *
 * \warning All methods and attributes are public and can be manipulated
 * directly from BRKGA algorithms. Note that this class is not meant to be used
 * externally of this unit.
 */
class Population {
public:
    /** \name Data members */
    //@{
    /// Population as vectors of probabilities.
    std::vector<Chromosome> population;

    /// Fitness (double) of a each chromosome.
    std::vector<std::pair<std::vector<double>, unsigned>> fitness;

    /// Number of non-dominated individuals.
    unsigned num_non_dominated;

    // Number of non-dominated fronts of individuals.
    unsigned num_fronts;

    /// Maximum number of elite individuals.
    unsigned max_num_elites;

    /// Number of elite individuals.
    unsigned num_elites;
    //@}

    /** \name Default constructors and destructor */
    //@{
    /**
     * \brief Default constructor.
     *
     * \param chr_size size of chromosome.
     * \param pop_size size of population.
     * \param max_num_elites maximum number of elite individuals.
     * \throw std::range_error if population size or chromosome size is zero.
     */
    Population(
            const unsigned chr_size,
            const unsigned pop_size,
            const unsigned max_num_elites_):
        population(pop_size, Chromosome(chr_size, 0.0)),
        fitness(pop_size),
        num_non_dominated(0),
        num_fronts(0),
        max_num_elites(max_num_elites_),
        num_elites(0)
    {
        if(pop_size == 0) {
            throw std::range_error("Population size cannot be zero.");
        }

        if(chr_size == 0) {
            throw std::range_error("Chromosome size cannot be zero.");
        }
    }

    /// Copy constructor.
    Population(const Population & other):
        population(other.population),
        fitness(other.fitness),
        num_non_dominated(other.num_non_dominated),
        num_fronts(other.num_non_dominated),
        max_num_elites(other.max_num_elites),
        num_elites(other.num_elites)
    {}

    /// Assignment operator for compliance.
    Population & operator=(const Population &) = default;

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
    unsigned getPopulationSize() const {
        return this->population.size();
    };

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
    double & operator()(const unsigned chromosome, const unsigned allele) {
        return this->population[chromosome][allele];
    }

    /**
     * \brief Returns a reference to a chromosome.
     * \param chromosome index of desired chromosome.
     * \returns a reference to chromosome.
     */
    Chromosome & operator()(unsigned chromosome) {
        return this->population[chromosome];
    }
    //@}

    /** \name Special access methods
     *
     * These methods REQUIRE fitness to be sorted, and thus a call to
     * `sortFitness()` beforehand.
     */
    //@{
    /// Returns the best fitnesses in this population.
    std::vector<std::vector<double>> getBestFitnesses(
            const std::vector<Sense> & senses) const {
        std::vector<std::vector<std::pair<std::vector<double>, unsigned>>>
            fronts = Population::nonDominatedSort<unsigned>(this->fitness,
                                                            senses);
        std::vector<std::vector<double>> result(fronts[0].size());
        std::transform(fronts[0].begin(),
                       fronts[0].end(),
                       result.begin(),
                       [](std::pair<std::vector<double>, unsigned> solution) {
                            return solution.first;
                       });
        return result;
    }

    /// Returns the best chromosomes in this population
    std::vector<Chromosome> getBestChromosomes(
            const std::vector<Sense> & senses) const {
        std::vector<std::vector<std::pair<std::vector<double>, unsigned>>>
            fronts = Population::nonDominatedSort<unsigned>(this->fitness,
                                                            senses);
        std::vector<Chromosome> result(fronts[0].size());

        for(unsigned i = 0; i < fronts[0].size(); i++) {
            result[i] = this->population[fronts[0][i].second];
        }

        return result;
    }

    /// Returns the fitness of chromosome i.
    std::vector<double> getFitness(const unsigned i)  const {
        return this->fitness[i].first;
    }

    /// Returns a reference to the i-th best chromosome.
    Chromosome & getChromosome(unsigned i) {
        return this->population[this->fitness[i].second];
    }

    /// Returns a const reference to the i-th best chromosome.
    const Chromosome & getChromosome(const unsigned i) const {
        return this->population[this->fitness[i].second];
    }

    //@}

    /** \name Other methods */
    //@{
    /**
     * \brief Returns `true` if `a1` is better than `a2`.
     *
     * This method depends on the optimization senses. When the optimization
     * sense is `Sense::MINIMIZE`, `a1 < a2` will return true, otherwise false.
     * The opposite happens for `Sense::MAXIMIZE`.
     */
    static inline bool betterThan(
            const std::vector<double> & a1, 
            const std::vector<double> & a2,
            const std::vector<Sense> & senses) {

        // checks if a1 is at least as good as a2
        for(unsigned i = 0; i < senses.size(); i++) {
            if(senses[i] == Sense::MINIMIZE) {
                if(a1[i] > a2[i] + std::numeric_limits<double>::epsilon()) {
                    return false;
                }
            } else {
                if(a1[i] < a2[i] - std::numeric_limits<double>::epsilon()) {
                    return false;
                }
            }
        }

        // checks if a1 is better than a2
        for(unsigned int i = 0; i < senses.size(); i++) {
            if(senses[i] == Sense::MINIMIZE) {
                if(a1[i] < a2[i] - std::numeric_limits<double>::epsilon()) {
                    return true;
                }
            } else {
                if(a1[i] > a2[i] + std::numeric_limits<double>::epsilon()) {
                    return true;
                }
            }
        }

        return false;
    }

    template <class T>
    static std::vector<std::vector<std::pair<std::vector<double>, T>>>
        nonDominatedSort(
            std::vector<std::pair<std::vector<double>, T>> fitness, 
            const std::vector<Sense> & senses) {

        std::vector<std::vector<std::pair<std::vector<double>, T>>> result;

        if(fitness.empty()) {
            return result;
        }

        auto comp = [senses](const std::pair<std::vector<double>, T> & a,
                             const std::pair<std::vector<double>, T> & b) ->
            bool {
            for(unsigned i = 0; i < a.first.size(); i++) {
                if(senses[i] == Sense::MINIMIZE) {
                    if(a.first[i] < b.first[i] -
                            std::numeric_limits<double>::epsilon()) {
                        return true;
                    }
                    if(a.first[i] > b.first[i] +
                            std::numeric_limits<double>::epsilon()) {
                        return false;
                    }
                } else { // senses[i] == Sense::MAXIMIZE
                    if(a.first[i] < b.first[i] -
                            std::numeric_limits<double>::epsilon()) {
                        return false;
                    }
                    if(a.first[i] > b.first[i] +
                            std::numeric_limits<double>::epsilon()) {
                        return true;
                    }
                }
            }
            // a == b
            return false;
        };

        std::sort(fitness.begin(), fitness.end(), comp);
        result.emplace_back(1, fitness.front());

        if(senses.size() == 1) {    
            for(unsigned i = 1; i < fitness.size(); i++) {
                if(Population::betterThan(fitness[i - 1].first,
                                          fitness[i].first, 
                                          senses)) {
                    result.emplace_back(1, fitness[i]);
                } else {
                    result.back().push_back(fitness[i]);
                }
            }
        } else { // senses.size() >= 2
            for(unsigned i = 1; i < fitness.size(); i++) {
                bool isDominated = false;

                // check if the current solution is dominated by a solution in
                // the last front
                for(unsigned j = result.back().size(); j > 0; j--) {
                    if(Population::betterThan(result.back()[j - 1].first,
                                              fitness[i].first,
                                              senses)) {
                        isDominated = true;
                        break;
                    }

                    // if there is only 2 objectives, we need to check for 
                    // dominance only with the last element in the last front
                    if(senses.size() == 2) {
                        break;
                    }
                }

                // if the current solution is dominated by a solution in the
                // last front
                if(isDominated) {
                    // create a new front to put the current solution
                    result.emplace_back(1, fitness[i]);
                } else {
                    // find the first front that does not have a solution that
                    // dominates the current solution using binary search
                    unsigned kMin = 0,
                             kMax = result.size();
                    while(kMin < kMax) {
                        unsigned k = floor((double(kMax) + double(kMin))/2.0);
                        isDominated = false;
                
                        // check if the current solution is dominated by a
                        // solution in the k-th front
                        for(unsigned j = result[k].size(); j > 0; j--) {
                            if(Population::betterThan(result[k][j - 1].first,
                                                      fitness[i].first,
                                                      senses)) {
                                isDominated = true;
                                break;
                            }

                            // if there is only 2 objectives, we need to check
                            // for dominance only with the last solution in the
                            // k-th front
                            if(senses.size() == 2) {
                                break;
                            }
                        }

                        if(isDominated) {
                            kMin = k + 1;
                        } else {
                            if(k == kMin) {
                                break;
                            }

                            kMax = k;
                        }
                    }

                    result[kMin].push_back(fitness[i]);
                }
            }
        }

        return result;
    }

    template <class T>
    static void crowdingSort(
            std::vector<std::pair<std::vector<double>, T>> & fitness) {
        std::vector<std::pair<double, std::pair<std::vector<double>, unsigned>>>
            aux(fitness.size());
        std::vector<double> distance(fitness.size(), 0.0);

        for(unsigned m = 0; m < fitness.front().first.size(); m++) {
            for(unsigned i = 0; i < aux.size(); i++) {
                aux[i] = std::make_pair(fitness[i].first[m],
                                        std::make_pair(fitness[i].first, i));
            }

            std::sort(aux.begin(), aux.end());

            double fMin = aux.front().first;
            double fMax = aux.back().first;

            distance[aux.front().second.second] =
                std::numeric_limits<double>::max();
            distance[aux.back().second.second] =
                std::numeric_limits<double>::max();

            for(unsigned i = 1; i < aux.size() - 1; i++) {
                if(distance[aux[i].second.second] < 
                        std::numeric_limits<double>::max()) {
                    distance[aux[i].second.second] += 
                        (aux[i + 1].second.first[m] - 
                         aux[i - 1].second.first[m]) / (fMax - fMin);
                }
            }
        }

        for(unsigned i = 0; i < aux.size(); i++) {
            aux[i].first = distance[aux[i].second.second];
        }

        std::sort(aux.begin(),
                  aux.end(),
                  std::greater<std::pair<double, 
                        std::pair<std::vector<double>, unsigned>>>());

        std::vector<std::pair<std::vector<double>, T>>
            sortedFitness(fitness.size());

        for(unsigned i = 0; i < fitness.size(); i++) {
            sortedFitness[i] = fitness[aux[i].second.second];
        }

        fitness = sortedFitness;
    }

    template <class T>
    static std::pair<unsigned, unsigned> sortFitness(
            std::vector<std::pair<std::vector<double>, T>> & fitness, 
            const std::vector<Sense> & senses) {
        if(senses.size() == 1) {
            std::sort(fitness.begin(),
                      fitness.end(),
                      [senses](const std::pair<std::vector<double>, T> & a,
                               const std::pair<std::vector<double>, T> & b){
                            if(senses.front() == Sense::MINIMIZE) {
                                return a.first.front() < b.first.front();
                            } else {
                                return a.first.front() > b.first.front();
                            }
                      });
            return std::make_pair(fitness.size(), 1);
        } else {
            std::vector<std::vector<std::pair<std::vector<double>, T>>> fronts =
                Population::nonDominatedSort<T>(fitness, senses);

            unsigned numSolutionsCopied = 0;
            for(unsigned f = 0; f < fronts.size(); f++) {
                Population::crowdingSort<T>(fronts[f]);
                std::copy(fronts[f].begin(), 
                          fronts[f].end(), 
                          fitness.begin() + numSolutionsCopied);
                numSolutionsCopied += fronts[f].size();
            }

            return std::make_pair(fronts.size(), fronts.front().size());
        }
    }

    /**
     * \brief Sorts `fitness` by its first parameter according to the senses.
     * \param senses Optimization senses.
     */
    void sortFitness(const std::vector<Sense> & senses) {
        auto ret = Population::sortFitness<unsigned>(this->fitness, senses);
        this->num_fronts = ret.first;
        this->num_non_dominated = ret.second;
        this->num_elites = this->num_non_dominated;

        if (this->num_elites > this->max_num_elites) {
            this->num_elites = this->max_num_elites;
        }
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
// BRKGA Params class.
//----------------------------------------------------------------------------//

/**
 * \brief Represents the BRKGA and IPR hyper-parameters.
 */
class BrkgaParams {
public:
    /** \name BRKGA Hyper-parameters */
    //@{
    /// Number of elements in the population.
    unsigned population_size;

    /// Maximum percentage of individuals to become the elite set.
    double max_elites_percentage;

    /// Mutation probability.
    double mutation_probability;

    /// Mutation distribution.
    double mutation_distribution;

    /// Number of elite parents for mating.
    unsigned num_elite_parents;

    /// Number of total parents for mating.
    unsigned total_parents;

    /// Type of bias that will be used.
    BiasFunctionType bias_type;

    /// Number of independent parallel populations.
    unsigned num_independent_populations;

    /// Number of incumbent solutions.
    unsigned num_incumbent_solutions;
    //@}

    /** \name Path Relinking parameters */
    //@{
    /// Number of pairs of chromosomes to be tested to path relinking.
    unsigned pr_number_pairs;

    /// Minimum distance between chromosomes selected to path-relinking.
    double pr_minimum_distance;

    /// Path relinking type.
    PathRelinking::Type pr_type;

    /// Individual selection to path-relinking.
    PathRelinking::Selection pr_selection;

    /// Defines the block size based on the size of the population.
    double alpha_block_size;

    /// Percentage / path size to be computed. Value in (0, 1].
    double pr_percentage;
    //@}

public:
    /** \name Default operators */
    //@{
    /// Default constructor.
    BrkgaParams():
        population_size(0),
        max_elites_percentage(0.0),
        mutation_probability(0.0),
        mutation_distribution(0.0),
        num_elite_parents(0),
        total_parents(0),
        bias_type(BiasFunctionType::CONSTANT),
        num_independent_populations(0),
        num_incumbent_solutions(0),
        pr_number_pairs(0),
        pr_minimum_distance(0.0),
        pr_type(PathRelinking::Type::DIRECT),
        pr_selection(PathRelinking::Selection::BESTSOLUTION),
        alpha_block_size(0.0),
        pr_percentage(0.0)
    {}

    /// Assignment operator for compliance.
    BrkgaParams & operator=(const BrkgaParams &) = default;

    /// Destructor.
    ~BrkgaParams() = default;
    //@}
};

//----------------------------------------------------------------------------//
// External Control Params class.
//----------------------------------------------------------------------------//

/**
 * \brief Represents additional control parameters that can be used outside this
 * framework.
 *
 * These parameters are not used directly in the BRKGA nor in the path
 * relinking. However, they are loaded from the configuration file and can be
 * called by the user to perform out-loop controlling.
 */
class ExternalControlParams {
public:
    /// Interval at which elite chromosomes are exchanged (0 means no exchange).
    unsigned exchange_interval;

    /// Number of elite chromosomes exchanged from each population.
    unsigned num_exchange_individuals;

    /// Interval at which the path relinking is applied 
    /// (0 means no path relinking).
    unsigned path_relink_interval;

    /// Interval at which the population are shaken (0 means no shaking).
    unsigned shake_interval;

    /// Interval at which the populations are reset (0 means no reset).
    unsigned reset_interval;

public:
    /** \name Default operators */
    //@{
    /// Default constructor.
    ExternalControlParams():
        exchange_interval(0),
        num_exchange_individuals(0),
        path_relink_interval(0),
        shake_interval(0),
        reset_interval(0)
    {}

    /// Assignment operator for compliance.
    ExternalControlParams & operator=(const ExternalControlParams &) = default;

    /// Destructor.
    ~ExternalControlParams() = default;
    //@}
};

//----------------------------------------------------------------------------//
// Loading the parameters from file
//----------------------------------------------------------------------------//

/**
 * \brief Reads the parameters from a configuration file.
 *
 * \param filename the configuration file.
 * \returns a tuple containing the BRKGA and external control parameters.
 * \throw std::fstream::failure in case of errors in the file.
 * \todo (ceandrade) This method can benefit from introspection tools
 *   from C++17. We would like achieve a code similar to the
 *   [Julia counterpart](<https://github.com/ceandrade/brkga_mp_ipr_julia>).
 */
INLINE std::pair<BrkgaParams, ExternalControlParams>
readConfiguration(const std::string & filename) {
    std::ifstream input(filename, std::ios::in);
    std::stringstream error_msg;

    if(!input) {
        error_msg << "File '" << filename << "' cannot be opened!";
        throw std::fstream::failure(error_msg.str());
    }

    std::unordered_map<std::string, bool> tokens({
        {"POPULATION_SIZE", false},
        {"MAX_ELITES_PERCENTAGE", false},
        {"MUTATION_PROBABILITY", false},
        {"MUTATION_DISTRIBUTION", false},
        {"NUM_ELITE_PARENTS", false},
        {"TOTAL_PARENTS", false},
        {"BIAS_TYPE", false},
        {"NUM_INDEPENDENT_POPULATIONS", false},
        {"NUM_INCUMBENT_SOLUTIONS", false},
        {"PR_NUMBER_PAIRS", false},
        {"PR_MINIMUM_DISTANCE", false},
        {"PR_TYPE", false},
        {"PR_SELECTION", false},
        {"ALPHA_BLOCK_SIZE", false},
        {"PR_PERCENTAGE", false},
        {"EXCHANGE_INTERVAL", false},
        {"NUM_EXCHANGE_INDIVIDUALS", false},
        {"PATH_RELINK_INTERVAL", false},
        {"SHAKE_INTERVAL", false},
        {"RESET_INTERVAL", false},
    });

    BrkgaParams brkga_params;
    ExternalControlParams control_params;

    std::string line;
    unsigned line_count = 0;

    while(std::getline(input, line)) {
        ++line_count;
        std::string::size_type pos = line.find_first_not_of(" \t\n\v");

        // Ignore all comments and blank lines.
        if(pos == std::string::npos || line[pos] == '#') {
            continue;
        }

        std::stringstream line_stream(line);
        std::string token, data;

        line_stream >> token >> data;

        std::transform(token.begin(), token.end(), token.begin(), toupper);
        if(tokens.find(token) == tokens.end()) {
            error_msg << "Invalid token on line " << line_count
                      << ": " << token;
            throw std::fstream::failure(error_msg.str());
        }

        if(tokens[token]) {
            error_msg << "Duplicate attribute on line " << line_count
                      << ": " << token << " already read!";
            throw std::fstream::failure(error_msg.str());
        }

        std::stringstream data_stream(data);
        bool fail = false;

        // TODO: for c++17, we may use std:any to short this code using a loop.
        if(token == "POPULATION_SIZE") {
            fail = !bool(data_stream >> brkga_params.population_size);
        } else if(token == "MAX_ELITES_PERCENTAGE") {
            fail = !bool(data_stream >> brkga_params.max_elites_percentage);
        } else if(token == "MUTATION_PROBABILITY") {
            fail = !bool(data_stream >> brkga_params.mutation_probability);
        } else if(token == "MUTATION_DISTRIBUTION") {
            fail = !bool(data_stream >> brkga_params.mutation_distribution);
        } else if(token == "NUM_ELITE_PARENTS") {
            fail = !bool(data_stream >> brkga_params.num_elite_parents);
        } else if(token == "TOTAL_PARENTS") {
            fail = !bool(data_stream >> brkga_params.total_parents);
        } else if(token == "BIAS_TYPE") {
            fail = !bool(data_stream >> brkga_params.bias_type);
        } else if(token == "NUM_INDEPENDENT_POPULATIONS") {
            fail = !bool(data_stream >>
                    brkga_params.num_independent_populations);
        } else if(token == "NUM_INCUMBENT_SOLUTIONS") {
            fail = !bool(data_stream >> brkga_params.num_incumbent_solutions);
        } else if(token == "PR_NUMBER_PAIRS") {
            fail = !bool(data_stream >> brkga_params.pr_number_pairs);
        } else if(token == "PR_MINIMUM_DISTANCE") {
            fail = !bool(data_stream >> brkga_params.pr_minimum_distance);
        } else if(token == "PR_TYPE") {
            fail = !bool(data_stream >> brkga_params.pr_type);
        } else if(token == "PR_SELECTION") {
            fail = !bool(data_stream >> brkga_params.pr_selection);
        } else if(token == "ALPHA_BLOCK_SIZE") {
            fail = !bool(data_stream >> brkga_params.alpha_block_size);
        } else if(token == "PR_PERCENTAGE") {
            fail = !bool(data_stream >> brkga_params.pr_percentage);
        } else if(token == "EXCHANGE_INTERVAL") {
            fail = !bool(data_stream >> control_params.exchange_interval);
        } else if(token == "NUM_EXCHANGE_INDIVIDUALS") {
            fail = !bool(data_stream >> 
                    control_params.num_exchange_individuals);
        } else if(token == "PATH_RELINK_INTERVAL") {
            fail = !bool(data_stream >> control_params.path_relink_interval);
        } else if(token == "SHAKE_INTERVAL") {
            fail = !bool(data_stream >> control_params.shake_interval);
        } else if(token == "RESET_INTERVAL") {
            fail = !bool(data_stream >> control_params.reset_interval);
        }

        if(fail) {
            error_msg << "Invalid value for '" << token
                      << "' on line "<< line_count
                      << ": '" << data << "'";
            throw std::fstream::failure(error_msg.str());
        }

        tokens[token] = true;
    }

    for(const auto & attribute_flag : tokens) {
        if(!attribute_flag.second) {
            error_msg << "Argument '" << attribute_flag.first
                      << "' was not supplied in the config file";
            throw std::fstream::failure(error_msg.str());
        }
    }

    return std::make_pair(std::move(brkga_params), std::move(control_params));
}

//----------------------------------------------------------------------------//
// Writing the parameters into file
//----------------------------------------------------------------------------//

/**
 * \brief Writes the parameters into a file..
 *
 * \param filename the configuration file.
 * \param brkga_params the BRKGA parameters.
 * \param control_params the external control parameters. Default is an empty
 *        object.
 * \throw std::fstream::failure in case of errors in the file.
 * \todo (ceandrade) This method can benefit from introspection tools
 *   from C++17. We would like achieve a code similar to the
 *   [Julia counterpart](<https://github.com/ceandrade/brkga_mp_ipr_julia>).
 */
INLINE void writeConfiguration(
        const std::string & filename,
        const BrkgaParams & brkga_params,
        const ExternalControlParams & control_params = ExternalControlParams()) {

    std::ofstream output(filename, std::ios::out);
    if(!output) {
        std::stringstream error_msg;
        error_msg << "File '" << filename << "' cannot be opened!";
        throw std::fstream::failure(error_msg.str());
    }

    output << "population_size " << brkga_params.population_size << std::endl
           << "max_elites_percentage " << brkga_params.max_elites_percentage 
           << std::endl
           << "mutation_probability " << brkga_params.mutation_probability
           << std::endl
           << "mutation_distribution " << brkga_params.mutation_distribution
           << std::endl
           << "num_elite_parents " << brkga_params.num_elite_parents
           << std::endl
           << "total_parents " << brkga_params.total_parents << std::endl
           << "bias_type " << brkga_params.bias_type << std::endl
           << "num_independent_populations "
           << brkga_params.num_independent_populations << std::endl
           << "num_incumbent_solutions " << brkga_params.num_incumbent_solutions
           << std::endl
           << "pr_number_pairs " << brkga_params.pr_number_pairs << std::endl
           << "pr_minimum_distance " << brkga_params.pr_minimum_distance
           << std::endl
           << "pr_type " << brkga_params.pr_type << std::endl
           << "pr_selection " << brkga_params.pr_selection << std::endl
           << "alpha_block_size " << brkga_params.alpha_block_size << std::endl
           << "pr_percentage " << brkga_params.pr_percentage << std::endl
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
 * \brief This class represents a Non-dominated Sorting Multi-Parent
 * Biased Random-key Genetic Algorithm (NSMPBRKGA).
 *
 * \author Carlos Eduardo de Andrade <ce.andrade@gmail.com>
 * \date 2020
 *
 * Main capabilities {#main_cap}
 * ========================
 *
 * Evolutionary process {#evol_process}
 * ------------------------
 *
 * In the NSMPBRKGA, we keep a population of chromosomes divided between the
 * elite and the non-elite group. During the mating, multiple parents are chosen
 * from the elite group and the non-elite group. They are sorted either on
 * no-decreasing order for minimization or non-increasing order to maximization
 * problems. Given this order, a bias function is applied to the rank of each
 * chromosome, resulting in weight for each one. Using a roulette method based
 * on the weights, the chromosomes are combined using a biased crossover.
 *
 * This code also implements the island model, where multiple populations
 * can be evolved in parallel, and migration between individuals between
 * the islands are performed using exchangeElite() method.
 *
 * This code requires the template argument `Decoder` be a class or functor
 * object capable to map a chromosome to a solution for the specific problem,
 * and return a value to be used as fitness to the decoded chromosome.
 * The decoder must have the method
 * \code{.cpp}
 *      double decode(Chromosome & chromosome, bool rewrite);
 * \endcode
 *
 * where #Chromosome is a `vector<double>` representing a solution and
 * `rewrite` is a boolean indicating that if the decode should rewrite the
 * chromosome in case it implements local searches and modifies the initial
 * solution decoded from the chromosome. Since this API has the capability of
 * decoding several chromosomes in parallel, the user must guarantee that
 * `Decoder::decode(...)` is thread-safe. Therefore, we do recommend to have
 * the writable variables per thread. Please, see the example that follows this
 * code.
 *
 * Implicit Path Relinking {#ipr}
 * ------------------------
 *
 * This API also implements the Implicit Path Relinking leveraging the decoder
 * capabilities. To perform the path relinking, the user must call pathRelink()
 * method, indicating the type of path relinking (direct or permutation-based,
 * see #PathRelinking::Type), the selection criteria (best solution or random
 * elite, see #PathRelinking::Selection), the distance function
 * (to choose individuals far enough, see BRKGA::DistanceFunctionBase,
 * BRKGA::HammingDistance, and BRKGA::KendallTauDistance), a maximum time or
 * a maximum path size.
 *
 * In the presence of multiple populations, the path relinking is performed
 * between elite chromosomes from different populations, in a circular fashion.
 * For example, suppose we have 3 populations. The framework performs 3 path
 * relinkings: the first between individuals from populations 1 and 2, the
 * second between populations 2 and 3, and the third between populations 3
 * and 1. In the case of just one population, both base and guiding individuals
 * are sampled from the elite set of that population.
 *
 * Note that the algorithm tries to find a pair of base and guiding solutions
 * with a minimum distance given by the distance function. If this is not
 * possible, a new pair of solutions are sampled (without replacement) and
 * tested against the distance. In case it is not possible to find such pairs
 * for the given populations, the algorithm skips to the next pair of
 * populations (in a circular fashion, as described above). Yet, if such pairs
 * are not found in any case, the algorithm declares failure. This indicates
 * that the populations are very homogeneous.
 *
 * The API will call `Decoder::decode()` always
 * with `rewrite = false`. The reason is that if the decoder rewrites the
 * chromosome, the path between solutions is lost and inadvertent results may
 * come up. Note that at the end of the path relinking, the method calls the
 * decoder with `rewrite = true` in the best chromosome found to guarantee
 * that this chromosome is re-written to reflect the best solution found.
 *
 * Other capabilities {#other_cap}
 * ========================
 *
 * Multi-start {#multi_start}
 * ------------------------
 *
 * This API also can be used as a simple multi-start algorithm without
 * evolution. To do that, the user must supply in the constructor the argument
 * `evolutionary_mechanism_on = false`. This argument makes the elite set has
 * one individual and the number of mutants n - 1, where n is the size of the
 * population. This setup disables the evolutionary process completely.
 *
 * Initial Population {#init_pop}
 * ------------------------
 *
 * This API allows the user provides a set of initial solutions to warm start
 * the algorithm. In general, initial solutions are created using other (fast)
 * heuristics and help the convergence of the BRKGA. To do that, the user must
 * encode the solutions on #Chromosome (`vector<double>`) and pass to the method
 * setInitialPopulation() as a `vector<#Chromosome>`.
 *
 * General Usage {#gen_usage}
 * ========================
 *
 *  -# The user must call the NSMPBRKGA constructor and pass the desired
 *     parameters. Please, see NSMPBRKGA::NSMPBRKGA for parameter details;
 *
 *      -# (Optional) The user provides the warm start solutions using
 *         setInitialPopulation();
 *
 *  -# The user must call the method initialize() to start the data structures
 *     and perform the decoding of the very first populations;
 *
 *  -# Main evolutionary loop:
 *
 *      -# On each iteration, the method evolve() should be called to perform
 *         the evolutionary step (or multi-steps if desired);
 *
 *      -# The user can check the current best chromosome (getBestChromosome())
 *         and its fitness (getBestFitness()) and perform checking and logging
 *         operations;
 *
 *      -# (Optional) The user can perform the individual migration between
 *         populations (exchangeElite());
 *
 *      -# (Optional) The user can perform the path relinking (pathRelink());
 *
 *      -# (Optional) The user can reset and start the algorithm over (reset());
 *
 * For a comprehensive and detailed usage, please see the examples that follow
 * this API.
 *
 * About multi-threading {#multi_thread}
 * ========================
 *
 * This API is capable of decoding several chromosomes in
 * parallel, as mentioned before. This capability is based on OpenMP
 * (<http://www.openmp.org>) and the compiler must have support to it.
 * Most recent versions of GNU G++ and Intel C++ Compiler support OpenMP.
 * Clang supports OpenMP since 3.8. However, there are some issues with the
 * libraries, and sometimes, the parallel sections are not enabled. On the
 * major, the user can find fixes to his/her system.
 *
 * Since, in general, the decoding process can be complex and lengthy, it is
 * recommended that **the number of threads used DO NOT exceed the number of
 * physical cores in the machine.** This improves the overall performance
 * drastically, avoiding cache misses and racing conditions. Note that the
 * number of threads is also tied to the memory utilization, and it should be
 * monitored carefully.
 *
 * History {#hist}
 * ========================
 *
 * This API was based on the code by Rodrigo Franco Toso, Sep 15, 2011.
 * http://github.com/rfrancotoso/brkgaAPI
 *
 */
template<class Decoder>
class NSMPBRKGA {
public:
    /** \name Constructors and destructor */
    //@{
    /**
     * \brief Builds the algorithm and its data structures with the given
     *        arguments.
     *
     * \param decoder_reference a reference to the decoder object. **NOTE:**
     *        BRKGA uses such object directly for decoding.
     * \param sense the optimization sense (maximization or minimization).
     * \param seed the seed for the random number generator.
     * \param chromosome_size number of genes in each chromosome.
     * \param params BRKGA and IPR parameters object loaded from a
     *        configuration file or manually created. All the data is copied.
     * \param max_threads number of threads to perform parallel decoding.\n
     *        **NOTE**: `Decoder::decode()` MUST be thread-safe.
     * \param evolutionary_mechanism_on if false, no evolution is performed
     *        but only chromosome decoding. Very useful to emulate a
     *        multi-start algorithm.
     *
     * \throw std::range_error if some parameter or combination of parameters
     *        does not fit.
     */
    NSMPBRKGA(
        Decoder & decoder_reference,
        const std::vector<Sense> senses,
        const unsigned seed,
        const unsigned chromosome_size,
        const BrkgaParams & params,
        const unsigned max_threads = 1,
        const bool evolutionary_mechanism_on = true);

    /// Destructor
    ~NSMPBRKGA() {}
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
     * \param chromosomes a set of individuals encoded as Chromosomes.
     * \throw std::runtime_error if the number of given chromosomes is larger
     *        than the population size; if the sizes of the given chromosomes
     *        do not match with the required chromosome size.
     */
    void setInitialPopulations(
            const std::vector<std::vector<Chromosome>> & populations);

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
    void setBiasCustomFunction(
            const std::function<double(const unsigned)> & func);

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
     *        NSMPBRKGAs.
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
     * \param pr_selection of which individuals use to path relinking.
     *        See PathRelinking::Selection.
     * \param dist a pointer to a functor/object to compute the distance between
     *        two chromosomes. This object must be inherited from
     *        BRKGA::DistanceFunctionBase and implement its methods.
     * \param number_pairs number of chromosome pairs to be tested.
     *        If 0, all pairs are tested.
     * \param minimum_distance between two chromosomes computed by `dist`.
     * \param block_size number of alleles to be exchanged at once in each
     *        iteration. If one, the traditional path relinking is performed.
     * \param max_time aborts path relinking when reach `max_time`.
     *        If `max_time <= 0`, no limit is imposed.
     * \param percentage defines the size, in percentage, of the path to
     *        build. Default: 1.0 (100%).
     *
     * \returns A PathRelinking::PathRelinkingResult depending on the relink
     *          status.
     *
     * \throw std::range_error if the percentage or size of the path is
     *        not in (0, 1].
     */
    PathRelinking::PathRelinkingResult pathRelink(
                    PathRelinking::Type pr_type,
                    PathRelinking::Selection pr_selection,
                    std::shared_ptr<DistanceFunctionBase> dist,
                    unsigned number_pairs,
                    double minimum_distance,
                    std::size_t block_size = 1,
                    long max_time = 0,
                    double percentage = 1.0);

    /**
     * \brief Performs path relinking between elite solutions that are,
     *        at least, a given minimum distance between themselves.
     *
     * This method uses all parameters supplied in the constructor.
     * In particular, the block size is computed by
     * \f$\lceil \alpha \times \sqrt{p} \rceil\f$
     * where \f$\alpha\f$ is BrkgaParams#alpha_block_size and
     * \f$p\f$ is BrkgaParams#population_size.
     * If the size is larger than the chromosome size, the size is set to
     * half of the chromosome size.
     *
     * Please, refer to #pathRelink() for details.
     *
     * \param dist a pointer to a functor/object to compute the distance between
     *        two chromosomes. This object must be inherited from
     *        BRKGA::DistanceFunctionBase and implement its methods.
     * \param max_time aborts path relinking when reach `max_time`.
     *        If `max_time <= 0`, no limit is imposed.
     *
     * \returns A PathRelinking::PathRelinkingResult depending on the relink
     *          status.
     *
     * \throw std::range_error if the percentage or size of the path is
     *        not in (0, 1].
     */
    PathRelinking::PathRelinkingResult pathRelink(
                    std::shared_ptr<DistanceFunctionBase> dist,
                    long max_time = 0);
    //@}

    /** \name Population manipulation methods */
    //@{
    /**
     * \brief Exchanges elite-solutions between the populations.

     * Given a population, the `num_immigrants` best solutions are copied to
     * the neighbor populations, replacing their worth solutions. If there is
     * only one population, nothing is done.

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
     * \throw std::runtime_error if the algorithm is not initialized.
     */
    void reset(double intensity = 1.0);

    /**
     * \brief Performs a shaking in the chosen population.
     * \param intensity the intensity of the shaking.
     * \param shaking_type either `CHANGE` or `SWAP` moves.
     * \param population_index the index of the population to be shaken. If
     * `population_index >= num_independent_populations`, all populations
     * are shaken.
     */
    void shake(unsigned intensity, 
               ShakingType shaking_type,
               unsigned population_index =
                    std::numeric_limits<unsigned>::infinity());

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
     * \param fitness the pre-computed fitness if it is available.
     *
     * \throw std::range_error either if `population_index` is larger
     *        than number of populations; or `position` is larger than the
     *        population size; or ` chromosome.size() != chromosome_size`
     */
    void injectChromosome(const Chromosome & chromosome,
                          unsigned population_index,
                          unsigned position,
                          std::vector<double> fitness = std::vector<double>(0));
    //@}

    /** \name Support methods */
    //@{
    /**
     * \brief Returns a reference to a current population.
     * \param population_index the population index.
     * \throw std::range_error if the index is larger than number of
     *        populations.
     */
    const Population & getCurrentPopulation(unsigned population_index = 0) const;

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
    const Chromosome & getChromosome(unsigned population_index,
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
    const BrkgaParams & getBrkgaParams() const {
        return this->params;
    }

    std::vector<Sense> getOptimizationSenses() const {
        return this->OPT_SENSES;
    }

    unsigned getChromosomeSize() const {
        return this->CHROMOSOME_SIZE;
    }

    unsigned getMaxNumElites() const {
        return this->max_num_elites;
    }

    bool evolutionaryIsMechanismOn() const {
        return this->evolutionary_mechanism_on;
    }

    unsigned getMaxThreads() const {
        return this->MAX_THREADS;
    }
    //@}

protected:
    /** \name BRKGA Hyper-parameters */
    //@{
    /// The BRKGA and IPR hyper-parameters.
    BrkgaParams params;

    /// Indicates whether we are maximizing or minimizing each objective.
    const std::vector<Sense> OPT_SENSES;

    /// Number of genes in the chromosome.
    const unsigned CHROMOSOME_SIZE;

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
    Decoder & decoder;

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

    /// Holds the sum of the results of each raking given a bias function.
    /// This value is needed to normalization.
    double total_bias_weight;

    /// Used to shuffled individual/chromosome indices during the mate.
    std::vector<unsigned> shuffled_individuals;

    /// Defines the order of parents during the mating.
    std::vector<std::pair<std::vector<double>, unsigned>> parents_ordered;

    /// Indicates if initial populations are set.
    bool initial_populations;

    /// Indicates if the algorithm was proper initialized.
    bool initialized;

    /// Indicates if the algorithm have been reset.
    bool reset_phase;

    /// Holds the start time for a call of the path relink procedure.
    std::chrono::system_clock::time_point pr_start_time;

    /// The current best solutions.
    std::vector<std::pair<std::vector<double>, Chromosome>> incumbentSolutions;
    //@}

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
    bool evolution(Population & curr, Population & next);

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
     * `O(chromosome_size^2 / block_size)` additional memory is necessary to
     * build the candidates, which can be costly if the `chromosome_size` is
     * very large.
     *
     * \param chr1 first chromosome.
     * \param chr2 second chromosome
     * \param dist distance functor (distance between two chromosomes).
     * \param[out] best_found best solution found in the search.
     * \param block_size number of alleles to be exchanged at once in each
     *        iteration. If one, the traditional path relinking is performed.
     * \param max_time abort path relinking when reach `max_time`.
     *        If `max_time <= 0`, no limit is imposed.
     * \param percentage define the size, in percentage, of the path to
     *        build. Default: 1.0 (100%).
     */
    bool directPathRelink(
            const Chromosome & chr1,
            const Chromosome & chr2,
            std::shared_ptr<DistanceFunctionBase> dist,
            std::pair<std::vector<double>, Chromosome> & best_found,
            std::size_t block_size,
            long max_time,
            double percentage);

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
     * `O(chromosome_size^2 / block_size)` additional memory is necessary to
     * build the candidates, which can be costly if the `chromosome_size` is
     * very large.
     *
     * The path relinking is performed by changing the order of
     * each allele of base chromosome for the correspondent one in
     * the guide chromosome.
     * \param chr1 first chromosome
     * \param chr2 second chromosome
     * \param dist distance functor (distance between two chromosomes)
     * \param[out] best_found best solution found in the search
     * \param block_size number of alleles to be exchanged at once in each
     *        iteration. If one, the traditional path relinking is performed.
     * \param max_time abort path relinking when reach `max_time`.
     *        If `max_time <= 0`, no limit is imposed.
     * \param percentage define the size, in percentage, of the path to
     *        build. Default: 1.0 (100%)
     */
    bool permutationBasedPathRelink(
            Chromosome & chr1, 
            Chromosome & chr2,
            std::shared_ptr<DistanceFunctionBase> dist,
            std::pair<std::vector<double>, Chromosome> & best_found,
            std::size_t block_size,
            long max_time,
            double percentage);

    bool updateIncumbentSolutions(
            std::vector<std::pair<std::vector<double>, Chromosome>> 
            newSolutions);
    //@}

    /** \name Helper functions */
    //@{
    /**
     * \brief Returns `true` if `a1` is better than `a2`.
     *
     * This method depends on the optimization senses. When the optimization
     * sense is `Sense::MINIMIZE`, `a1 < a2` will return true, otherwise false.
     * The opposite happens for `Sense::MAXIMIZE`.
     */
    inline bool betterThan(const std::vector<double> a1, 
                           const std::vector<double> a2) const;

    /// Distributes real values of given precision across [0, 1] evenly.
    inline double rand01();

    /// Returns a number between `0` and `n`.
    inline uint_fast32_t randInt(const uint_fast32_t n);
    //@}
};

//----------------------------------------------------------------------------//

template<class Decoder>
NSMPBRKGA<Decoder>::NSMPBRKGA(
        Decoder & _decoder_reference,
        const std::vector<Sense> _senses,
        unsigned _seed,
        unsigned _chromosome_size,
        const BrkgaParams & _params,
        const unsigned _max_threads,
        const bool _evolutionary_mechanism_on):

        // Algorithm parameters.
        params(_params),
        OPT_SENSES(_senses),
        CHROMOSOME_SIZE(_chromosome_size),
        max_num_elites(_evolutionary_mechanism_on ?
                       unsigned(params.max_elites_percentage *
                                params.population_size)
                       : 1),
        evolutionary_mechanism_on(_evolutionary_mechanism_on),
        MAX_THREADS(_max_threads),

        // Internal data.
        decoder(_decoder_reference),
        rng(_seed),
        previous(params.num_independent_populations, nullptr),
        current(params.num_independent_populations, nullptr),
        bias_function(),
        total_bias_weight(0.0),
        shuffled_individuals(params.population_size),
        parents_ordered(params.total_parents),
        initial_populations(false),
        initialized(false),
        reset_phase(false),
        pr_start_time(),
        incumbentSolutions()
{
    using std::range_error;
    std::stringstream ss;

    if(this->CHROMOSOME_SIZE == 0) {
        ss << "Chromosome size must be larger than zero: " 
           << this->CHROMOSOME_SIZE;
    } else if(this->params.population_size == 0) {
        ss << "Population size must be larger than zero: " 
           << this->params.population_size;
    } else if(this->params.mutation_probability <=
            std::numeric_limits<double>::epsilon()) {
        ss << "Mutation probability (" << this->params.mutation_probability
           << ") smaller or equal to zero.";
    } else if(this->params.mutation_distribution <=
            std::numeric_limits<double>::epsilon()) {
        ss << "Mutation distribution (" << this->params.mutation_distribution
           << ") smaller or equal to zero.";
    } else if(this->max_num_elites > this->params.population_size) {
        ss << "Maximum elite-set size (" << this->max_num_elites
           << ") greater than population size (" 
           << this->params.population_size << ")";
    } else if(this->params.num_elite_parents < 1) {
        ss << "num_elite_parents must be at least 1: " 
           << this->params.num_elite_parents;
    } else if(this->params.total_parents < 2) {
        ss << "Total_parents must be at least 2: " 
           << this->params.total_parents;
    } else if(this->params.num_elite_parents >= this->params.total_parents) {
        ss << "Num_elite_parents (" << this->params.num_elite_parents << ") "
           << "is greater than total_parents (" 
           << this->params.total_parents << ")";
    } else if(this->params.num_independent_populations == 0) {
        ss << "Number of parallel populations cannot be zero.";
    } else if(this->params.alpha_block_size < 1e-6) {
        ss << "(alpha) block size <= 0.0";
    } else if(this->params.pr_percentage < 1e-6 
           || this->params.pr_percentage > 1.0) {
        ss << "Path relinking percentage (" << this->params.pr_percentage
           << ") is not in the range (0, 1].";
    }

    const auto str_error = ss.str();
    if(str_error.length() > 0) {
        throw range_error(str_error);
    }

    // Chooses the bias function.
    switch(this->params.bias_type) {
        case BiasFunctionType::LOGINVERSE: {
            // Same as log(r + 1), but avoids precision loss.
            this->setBiasCustomFunction(
                [](const unsigned r) { return 1.0 / log1p(r); }
            );
            break;
        }

        case BiasFunctionType::LINEAR: {
            this->setBiasCustomFunction(
                [](const unsigned r) { return 1.0 / r; }
            );
            break;
        }

        case BiasFunctionType::QUADRATIC: {
            this->setBiasCustomFunction(
                [](const unsigned r) { return 1.0 / (r * r); }
            );
            break;
        }

        case BiasFunctionType::CUBIC: {
            this->setBiasCustomFunction(
                [](const unsigned r) { return 1.0 / (r * r * r); }
            );
            break;
        }

        case BiasFunctionType::EXPONENTIAL: {
            this->setBiasCustomFunction(
                [](const unsigned r) { return exp(-1.0 * r); }
            );
            break;
        }

        case BiasFunctionType::CONSTANT:
        default: {
            this->setBiasCustomFunction(
                [&](const unsigned) { return 1.0 / this->params.total_parents; }
            );
            break;
        }
    }

    this->rng.discard(1000);  // Discard some states to warm up.
}

//----------------------------------------------------------------------------//

template<class Decoder>
inline bool NSMPBRKGA<Decoder>::betterThan(
        const std::vector<double> a1, 
        const std::vector<double> a2) const {
    return Population::betterThan(a1, a2, this->OPT_SENSES);
}

//----------------------------------------------------------------------------//

template<class Decoder>
const Population &
NSMPBRKGA<Decoder>::getCurrentPopulation(unsigned population_index) const {
    if(population_index >= this->current.size()) {
        throw std::range_error("The index is larger than number of "
                               "populations");
    }
    return (*(this->current)[population_index]);
}

//----------------------------------------------------------------------------//

template<class Decoder>
std::vector<std::vector<double>>
NSMPBRKGA<Decoder>::getIncumbentFitnesses() const {
    std::vector<std::vector<double>> result;

    for(unsigned i = 0; i < this->incumbentSolutions.size(); i++) {
        result.push_back(incumbentSolutions[i].first);
    }

    return result;
}

//----------------------------------------------------------------------------//

template<class Decoder>
std::vector<Chromosome> 
NSMPBRKGA<Decoder>::getIncumbentChromosomes() const {
    std::vector<Chromosome> result;

    for(unsigned i = 0; i < this->incumbentSolutions.size(); i++) {
        result.push_back(this->incumbentSolutions[i].second);
    }

    return result;
}

//----------------------------------------------------------------------------//

template<class Decoder>
const std::vector<std::pair<std::vector<double>, Chromosome>> &
NSMPBRKGA<Decoder>::getIncumbentSolutions() const {
    return this->incumbentSolutions;
}

//----------------------------------------------------------------------------//

template<class Decoder>
std::vector<double> NSMPBRKGA<Decoder>::getFitness(
        unsigned population_index, 
        unsigned position) const {
    if(population_index >= this->current.size()) {
        throw std::range_error("The population index is larger than number of "
                               "populations");
    }

    if(position >= this->params.population_size) {
        throw std::range_error("The chromosome position is larger than number "
                               "of populations");
    }

    return this->current[population_index]->fitness[position].first;
}

//----------------------------------------------------------------------------//

template<class Decoder>
const Chromosome & NSMPBRKGA<Decoder>::getChromosome(
        unsigned population_index, 
        unsigned position) const {
    if(population_index >= this->current.size()) {
        throw std::range_error("The population index is larger than number of "
                               "populations");
    }

    if(position >= this->params.population_size) {
        throw std::range_error("The chromosome position is larger than number "
                               "of populations");
    }

    return this->current[population_index]->getChromosome(position);
}

//----------------------------------------------------------------------------//

template<class Decoder>
void NSMPBRKGA<Decoder>::injectChromosome(const Chromosome & chromosome,
                                          unsigned population_index,
                                          unsigned position,
                                          std::vector<double> fitness) {
    if(population_index >= this->current.size()) {
        throw std::range_error("The population index is larger than number of "
                               "populations");
    }

    if(position >= this->params.population_size) {
        throw std::range_error("The chromosome position is larger than number "
                               "of populations");
    }

    if(chromosome.size() != this->CHROMOSOME_SIZE) {
        throw std::range_error("Wrong chromosome size");
    }

    auto & pop = this->current[population_index];
    auto & local_chr = pop->population[pop->fitness[position].second];
    local_chr = chromosome;

    if(!(fitness.size() > 0)) {
        fitness = this->decoder.decode(local_chr, true);
    }

    pop->setFitness(position, fitness);
    pop->sortFitness(this->OPT_SENSES);

    this->updateIncumbentSolutions(
            std::vector<std::pair<std::vector<double>, Chromosome>>(
                1, 
                std::make_pair(fitness, chromosome)));
}

//----------------------------------------------------------------------------//

template<class Decoder>
void NSMPBRKGA<Decoder>::setBiasCustomFunction(
        const std::function<double(const unsigned)> & func) {

    std::vector<double> bias_values(this->params.total_parents);
    std::iota(bias_values.begin(), bias_values.end(), 1);
    std::transform(bias_values.begin(), bias_values.end(),
                   bias_values.begin(), func);

    // If it is not non-increasing, throw an error.
    if(!std::is_sorted(bias_values.rbegin(), bias_values.rend())) {
        throw std::runtime_error("bias_function must be positive "
                                 "non-decreasing");
    }

    if(this->bias_function) {
        this->params.bias_type = BiasFunctionType::CUSTOM;
    }

    this->bias_function = func;
    this->total_bias_weight = std::accumulate(bias_values.begin(),
                                              bias_values.end(),
                                              0.0);
}

//----------------------------------------------------------------------------//

template<class Decoder>
void NSMPBRKGA<Decoder>::reset(double intensity) {
    if(!this->initialized) {
        throw std::runtime_error("The algorithm hasn't been initialized. "
                                 "Don't forget to call initialize() method");
    }

    this->reset_phase = true;

    unsigned total_individuals_reset = intensity * this->params.population_size;

    // Initialize each chromosome of the current population.
    for(unsigned i = 0; i < this->params.num_independent_populations; i++) {
        // Rebuild the indices.
        std::iota(this->shuffled_individuals.begin(), 
                  this->shuffled_individuals.end(),
                  0);
        // Shuffles individuals.
        std::shuffle(this->shuffled_individuals.begin(),
                     this->shuffled_individuals.end(),
                     this->rng);
        for(unsigned j = 0; j < total_individuals_reset; j++) {
            for(unsigned k = 0; k < this->CHROMOSOME_SIZE; k++) {
                (*this->current[i])(shuffled_individuals[j], k) =
                    this->rand01();
            }
        }
    }

    std::vector<std::pair<std::vector<double>, Chromosome>>
        newSolutions(this->params.num_independent_populations *
                this->params.population_size);

    // Initialize and decode each chromosome of the current population,
    // then copy to previous.
    for(unsigned i = 0; i < this->params.num_independent_populations; ++i) {
        #ifdef _OPENMP
            #pragma omp parallel for num_threads(MAX_THREADS) schedule(static,1)
        #endif
        for(unsigned j = 0; j < this->params.population_size; ++j) {
            this->current[i]->setFitness(j,
                    this->decoder.decode((*this->current[i])(j), true));
            newSolutions[i * this->params.population_size + j] =
                std::make_pair(this->current[i]->getFitness(j), 
                               (*this->current[i])(j));
        }

        // Sort and copy to previous.
        this->current[i]->sortFitness(this->OPT_SENSES);
    }

    this->updateIncumbentSolutions(newSolutions);

    this->reset_phase = false;
}

//----------------------------------------------------------------------------//

template<class Decoder>
bool NSMPBRKGA<Decoder>::evolve(unsigned generations) {
    if(!this->initialized) {
        throw std::runtime_error("The algorithm hasn't been initialized. "
                                 "Don't forget to call initialize() method");
    }

    if(generations == 0) {
        throw std::range_error("Cannot evolve for 0 generations.");
    }

    bool result = false;

    for(unsigned i = 0; i < generations; ++i) {
        for(unsigned j = 0; j < this->params.num_independent_populations; ++j) {
            // First evolve the population (current, next).
            if(this->evolution(*(this->current)[j], *(this->previous)[j])) {
                result = true;
            }

            std::swap(this->current[j], this->previous[j]);
        }
    }

    return result;
}

//----------------------------------------------------------------------------//

template<class Decoder>
void NSMPBRKGA<Decoder>::exchangeElite(unsigned num_immigrants) {
    if(this->params.num_independent_populations == 1) {
        return;
    }

    unsigned immigrants_threshold = ceil(this->params.population_size /
            (this->params.num_independent_populations - 1));

    if(num_immigrants < 1 || num_immigrants >= immigrants_threshold) {
        std::stringstream ss;
        ss << "Number of immigrants (" << num_immigrants << ") less than one, "
              "or larger than or equal to population size / "
              "num_independent_populations (" << immigrants_threshold << ")";
        throw std::range_error(ss.str());
    }

    #ifdef _OPENMP
        #pragma omp parallel for num_threads(MAX_THREADS)
    #endif
    for(unsigned i = 0; i < this->params.num_independent_populations; ++i) {
        // Population i will receive some elite members from each Population j.
        // Last chromosome of i (will be overwritten below).
        unsigned dest = this->params.population_size - 1;
        for(unsigned j = 0; j < this->params.num_independent_populations; ++j) {
            if(j == i) {
                continue;
            }

            // Copy the num_immigrants best from Population j into Population i.
            for(unsigned m = 0; m < num_immigrants; ++m) {
                // Copy the m-th best of Population j into the 'dest'-th
                // position of Population i
                const auto best_of_j = this->current[j]->getChromosome(m);
                std::copy(best_of_j.begin(), best_of_j.end(),
                          this->current[i]->getChromosome(dest).begin());
                this->current[i]->fitness[dest].first = 
                    this->current[j]->fitness[m].first;
                --dest;
            }
        }
    }

    // Re-sort each population since they were modified.
    #ifdef _OPENMP
        #pragma omp parallel for num_threads(MAX_THREADS)
    #endif
    for(unsigned i = 0; i < this->params.num_independent_populations; ++i) {
        this->current[i]->sortFitness(this->OPT_SENSES);
    }
}

//----------------------------------------------------------------------------//

template<class Decoder>
void NSMPBRKGA<Decoder>::setInitialPopulations(
        const std::vector<std::vector<Chromosome>> & populations) {
    if(populations.size() > this->params.num_independent_populations) {
        std::stringstream ss;
        ss << "Number of given populations (" << populations.size() << ") is "
           << "larger than the maximum number of independent populations (" 
           << this->params.num_independent_populations<<")";
        throw std::runtime_error(ss.str());
    }

    for(unsigned i = 0; i < populations.size(); i++) {
        std::vector<Chromosome> chromosomes = populations[i];

        if(chromosomes.size() > this->params.population_size) {
            std::stringstream ss;
            ss << "Error on setting initial population " << i << ": number of "
               << "given chromosomes (" << chromosomes.size() << ") is larger "
               << "than the population size (" << this->params.population_size
               << ")";
            throw std::runtime_error(ss.str());
        }

        this->current[i].reset(new Population(this->CHROMOSOME_SIZE, 
                                              chromosomes.size(),
                                              this->max_num_elites));

        for(unsigned j = 0; j < chromosomes.size(); j++) {
            Chromosome chr = chromosomes[j];

            if(chr.size() != this->CHROMOSOME_SIZE) {
                std::stringstream ss;
                ss << "Error on setting initial population " << i << ": "
                   << "chromosome " << j << " does not have the required "
                   << "dimension (actual size: " << chr.size() << ", required "
                   << "size: " << this->CHROMOSOME_SIZE << ")";
                throw std::runtime_error(ss.str());
            }

            std::copy(chr.begin(), 
                      chr.end(),
                      this->current[i]->population[j].begin());
        }
    }

    this->initial_populations = true;
}

//----------------------------------------------------------------------------//

template<class Decoder>
void NSMPBRKGA<Decoder>::initialize() {
    // Verify the initial population and complete or prune it!
    if(this->initial_populations) {
        for(unsigned i = 0; i < this->params.num_independent_populations; i++) {
            if(this->current[i]->population.size() <
                    this->params.population_size) {
                auto pop = this->current[i];
                Chromosome chromosome(this->CHROMOSOME_SIZE);
                unsigned j = pop->population.size();

                pop->population.resize(this->params.population_size);
                pop->fitness.resize(this->params.population_size);

                for(; j < this->params.population_size; j++) {
                    for(unsigned k = 0; k < this->CHROMOSOME_SIZE; ++k) {
                        chromosome[k] = this->rand01();
                    }

                    pop->population[j] = chromosome;
                }
            }
            // Prune some additional chromosomes.
            else if(this->current[i]->population.size() >
                    this->params.population_size) {
                this->current[i]->population.resize(this->params.population_size);
                this->current[i]->fitness.resize(this->params.population_size);
            }
        }
    } else {
        // Initialize each chromosome of the current population.
        for(unsigned i = 0; i < this->params.num_independent_populations; i++) {
            if(!this->reset_phase) {
                this->current[i].reset(
                        new Population(this->CHROMOSOME_SIZE,
                                       this->params.population_size,
                                       this->max_num_elites));
            }

            for(unsigned j = 0; j < this->params.population_size; j++) {
                for(unsigned k = 0; k < this->CHROMOSOME_SIZE; k++) {
                    (*this->current[i])(j, k) = this->rand01();
                }
            }
        }
    }

    std::vector<std::pair<std::vector<double>, Chromosome>>
        newSolutions(this->params.num_independent_populations *
                this->params.population_size);

    // Initialize and decode each chromosome of the current population,
    // then copy to previous.
    for(unsigned i = 0; i < this->params.num_independent_populations; ++i) {
        #ifdef _OPENMP
            #pragma omp parallel for num_threads(MAX_THREADS) schedule(static,1)
        #endif
        for(unsigned j = 0; j < this->params.population_size; ++j) {
            this->current[i]->setFitness(j,
                    this->decoder.decode((*this->current[i])(j), true));
            newSolutions[i * this->params.population_size + j] =
                std::make_pair(this->current[i]->getFitness(j), 
                               (*this->current[i])(j));
        }

        // Sort and copy to previous.
        this->current[i]->sortFitness(this->OPT_SENSES);

        this->previous[i].reset(new Population(*this->current[i]));
    }

    this->updateIncumbentSolutions(newSolutions);

    this->initialized = true;
}

//----------------------------------------------------------------------------//

template<class Decoder>
void NSMPBRKGA<Decoder>::shake(unsigned intensity,
                               ShakingType shaking_type,
                               unsigned population_index) {
    if(!this->initialized) {
        throw std::runtime_error("The algorithm hasn't been initialized. "
                                 "Don't forget to call initialize() method");
    }

    unsigned pop_start = population_index;
    unsigned pop_end = population_index;
    if(population_index >= this->params.num_independent_populations) {
        pop_start = 0;
        pop_end = this->params.num_independent_populations - 1;
    }

    for(; pop_start <= pop_end; ++pop_start) {
        auto& pop = this->current[pop_start]->population;

        // Shake the elite set.
        for(unsigned e = 0; e < this->current[pop_start]->num_elites; ++e) {
            for(unsigned k = 0; k < intensity; ++k) {
                auto i = this->randInt(this->CHROMOSOME_SIZE - 2);
                if(shaking_type == ShakingType::CHANGE) {
                    // Invert value.
                    pop[e][i] = 1.0 - pop[e][i];
                } else {
                    // Swap with neighbor.
                    std::swap(pop[e][i], pop[e][i + 1]);
                }

                i = this->randInt(this->CHROMOSOME_SIZE - 1);
                if(shaking_type == ShakingType::CHANGE) {
                    // Change to random value.
                    pop[e][i] = this->rand01();
                } else {
                    // Swap two random positions.
                    auto j = this->randInt(this->CHROMOSOME_SIZE - 1);
                    std::swap(pop[e][i], pop[e][j]);
                }
            }
        }

        // Reset the remaining population.
        for(unsigned ne = this->current[pop_start]->num_elites;
                ne < this->params.population_size; ++ne) {
            for(unsigned k = 0; k < this->CHROMOSOME_SIZE; ++k) {
                pop[ne][k] = this->rand01();
            }
        }

        std::vector<std::pair<std::vector<double>, Chromosome>>
            newSolutions(this->params.population_size);

        #ifdef _OPENMP
            #pragma omp parallel for num_threads(MAX_THREADS) schedule(static,1)
        #endif
        for(unsigned j = 0; j < this->params.population_size; ++j) {
            this->current[pop_start]->setFitness(j,
                    this->decoder.decode((*this->current[pop_start])(j), true));
            newSolutions[j] =
                std::make_pair(this->current[pop_start]->getFitness(j),
                        (*this->current[pop_start])(j));
        }

        this->updateIncumbentSolutions(newSolutions);

        // Now we must sort by fitness, since things might have changed.
        this->current[pop_start]->sortFitness(this->OPT_SENSES);
    }
}

//----------------------------------------------------------------------------//

template<class Decoder>
bool NSMPBRKGA<Decoder>::evolution(Population & curr,
                                   Population & next) {
    bool result = false;

    // First, we copy the elite chromosomes to the next generation.
    for(unsigned chr = 0; chr < curr.num_elites; ++chr) {
        next.population[chr] = curr.population[curr.fitness[chr].second];
        next.fitness[chr] = std::make_pair(curr.fitness[chr].first, chr);
    }

    // Second, we generate 'num_objectives' offspring,
    // always using one of the best individuals.
    for(unsigned chr = curr.num_elites;
            chr < curr.num_elites + this->OPT_SENSES.size(); ++chr) {
        // First take one of the best individuals
        // Then we shuffled the elite set indices, and take the elite parents.
        // Then we shuffle all indices and take the remaining parents.

        this->parents_ordered.clear();

        // Take one of the best individuals.
        this->parents_ordered.emplace_back(curr.fitness[chr - curr.num_elites]);

        // Rebuild the indices.
        std::iota(this->shuffled_individuals.begin(), 
                  this->shuffled_individuals.end(), 
                  0);

        // Shuffles elite.
        std::shuffle(this->shuffled_individuals.begin(),
                     this->shuffled_individuals.begin() + curr.num_elites,
                     this->rng);

        // Take the elite parents.
        for(unsigned j = 0; j < params.num_elite_parents - 1; j++) {
            this->parents_ordered.emplace_back(
                    curr.fitness[shuffled_individuals[j]]);
        }

        // Shuffles whole population
        std::shuffle(this->shuffled_individuals.begin(),
                     this->shuffled_individuals.end(),
                     this->rng);

        // Take the remaining parents.
        for(unsigned j = 0; j < this->params.total_parents -
                this->params.num_elite_parents; ++j) {
            this->parents_ordered.emplace_back(
                    curr.fitness[shuffled_individuals[j]]);
        }

        // Sort parents
        Population::sortFitness<unsigned>(this->parents_ordered,
                                          this->OPT_SENSES);

        // Performs the mate.
        for(unsigned allele = 0; allele < this->CHROMOSOME_SIZE; ++allele) {
            // Roulette method.
            unsigned parent = 0;
            double cumulative_probability = 0.0;
            const double toss = this->rand01();
            do {
                // Start parent from 1 because the bias function.
                cumulative_probability += this->bias_function(++parent) /
                                          this->total_bias_weight;
            } while(cumulative_probability < toss);

            // Decrement parent to the right index, and take the allele value.
            next(chr, allele) = curr(this->parents_ordered[--parent].second, 
                                     allele);
        }
    }

    // Third, we generate 'pop_size - num_objectives - num_elites' offspring.
    for(unsigned chr = curr.num_elites + this->OPT_SENSES.size();
        chr < this->params.population_size; 
        ++chr) {
        // First, we shuffled the elite set indices, and take the elite parents.
        // Then we shuffle all indices and take the remaining parents.

        this->parents_ordered.clear();

        // Rebuild the indices.
        std::iota(this->shuffled_individuals.begin(), 
                  this->shuffled_individuals.end(),
                  0);

        // Shuffles elite.
        std::shuffle(this->shuffled_individuals.begin(),
                     this->shuffled_individuals.begin() + curr.num_elites,
                     this->rng);

        // Take the elite parents.
        for(unsigned j = 0; j < params.num_elite_parents; j++) {
            this->parents_ordered.emplace_back(
                    curr.fitness[shuffled_individuals[j]]);
        }

        // Shuffles whole population
        std::shuffle(this->shuffled_individuals.begin(),
                     this->shuffled_individuals.end(),
                     this->rng);

        // Take the remaining parents.
        for(unsigned j = 0; j < this->params.total_parents -
                this->params.num_elite_parents; ++j) {
            this->parents_ordered.emplace_back(
                    curr.fitness[shuffled_individuals[j]]);
        }

        // Sort parents
        Population::sortFitness<unsigned>(this->parents_ordered,
                                          this->OPT_SENSES);

        // Performs the mate.
        for(unsigned allele = 0; allele < this->CHROMOSOME_SIZE; ++allele) {
            // Roulette method.
            unsigned parent = 0;
            double cumulative_probability = 0.0;
            const double toss = this->rand01();
            do {
                // Start parent from 1 because the bias function.
                cumulative_probability += this->bias_function(++parent) /
                                          this->total_bias_weight;
            } while(cumulative_probability < toss);

            // Decrement parent to the right index, and take the allele value.
            next(chr, allele) = curr(this->parents_ordered[--parent].second, 
                                     allele);
        }

        // Performs polynomial mutation.
        for(unsigned allele = 0; allele < this->CHROMOSOME_SIZE; ++allele) {
            if(this->rand01() < this->params.mutation_probability) {
                double y = next(chr, allele),
                       val = std::pow(1 - std::min(y, 1.0 - y),
                                      this->params.mutation_distribution + 1.0),
                       exponent = 1.0 / 
                           (this->params.mutation_distribution + 1.0),
                       delta_q = 0.0,
                       u = this->rand01();

                if(u <= 0.5) {
                    delta_q = std::pow(2.0 * u + (1.0 - 2.0 * u) * val,
                            exponent) - 1.0;
                } else {
                    delta_q = 1.0 - std::pow(2.0 * (1.0 - u) + 2.0 * (u - 0.5) *
                            val, exponent);
                }

                next(chr, allele) += delta_q;
            }
        }
    }

    // To finish, we fill up the remaining spots with mutants.
//    for(unsigned chr = this->params.population_size - curr.num_mutants;
//            chr < this->params.population_size; ++chr) {
//        for(auto & allele : next.population[chr]) {
//            allele = this->rand01();
//        }
//    }

    std::vector<std::pair<std::vector<double>, Chromosome>>
        newSolutions(this->params.population_size - curr.num_elites);

    // Time to compute fitness, in parallel.
    #ifdef _OPENMP
        #pragma omp parallel for num_threads(MAX_THREADS) schedule(static, 1)
    #endif
    for(unsigned i = curr.num_elites; i < this->params.population_size; ++i) {
        next.setFitness(i, this->decoder.decode(next.population[i], true));
        newSolutions[i - curr.num_elites] =
            std::make_pair(next.fitness[i].first, next.population[i]);
    }

    if(this->updateIncumbentSolutions(newSolutions)) {
        result = true;
    }

    // Now we must sort by fitness, since things might have changed.
    next.sortFitness(this->OPT_SENSES);

    return result;
}

//----------------------------------------------------------------------------//

template<class Decoder>
PathRelinking::PathRelinkingResult NSMPBRKGA<Decoder>::pathRelink(
                    PathRelinking::Type pr_type,
                    PathRelinking::Selection pr_selection,
                    std::shared_ptr<DistanceFunctionBase> dist,
                    unsigned number_pairs,
                    double minimum_distance,
                    std::size_t block_size,
                    long max_time,
                    double percentage) {

    using PR = PathRelinking::PathRelinkingResult;

    if(percentage < 1e-6 || percentage > 1.0) {
        throw std::range_error("Percentage/size of path relinking invalid.");
    }

    if(max_time <= 0) {
        max_time = std::numeric_limits<long>::max();
    }

    Chromosome initial_solution(this->CHROMOSOME_SIZE);
    Chromosome guiding_solution(this->CHROMOSOME_SIZE);

    // Perform path relinking between elite chromosomes from different
    // populations. This is done in a circular fashion.

    std::deque<std::pair<std::size_t, std::size_t>> index_pairs;

    // Keep track of the time.
    this->pr_start_time = std::chrono::system_clock::now();

    auto final_status = PR::TOO_HOMOGENEOUS;

    for(unsigned pop_count = 0; pop_count <
            this->params.num_independent_populations; ++pop_count) {
        auto elapsed_seconds =
            std::chrono::duration_cast<std::chrono::seconds>
            (std::chrono::system_clock::now() - this->pr_start_time).count();
        if(elapsed_seconds > max_time) {
            break;
        }

        unsigned pop_base = pop_count;
        unsigned pop_guide = pop_count + 1;
        bool found_pair = false;

        // If we have just one population, we take the both solution from it.
        if(this->params.num_independent_populations == 1) {
            pop_base = pop_guide = 0;
            pop_count = this->params.num_independent_populations;
        }
        // If we have two populations, perform just one path relinking.
        else if(this->params.num_independent_populations == 2) {
            pop_count = this->params.num_independent_populations;
        }

        // Do the circular thing.
        if(pop_guide == this->params.num_independent_populations) {
            pop_guide = 0;
        }

        const unsigned num_elites = this->current[pop_base]->num_elites;

        index_pairs.clear();
        for(std::size_t i = 0; i < num_elites; ++i) {
            for(std::size_t j = 0; j < num_elites; ++j) {
                index_pairs.emplace_back(std::make_pair(i, j));
            }
        }

        unsigned tested_pairs_count = 0;
        if(number_pairs == 0) {
            number_pairs = index_pairs.size();
        }

        while(!index_pairs.empty() && tested_pairs_count < number_pairs &&
              elapsed_seconds < max_time) {
            const auto index =
                    (pr_selection == PathRelinking::Selection::BESTSOLUTION?
                     0 : this->randInt(index_pairs.size() - 1));

            const auto pos1 = index_pairs[index].first;
            const auto pos2 = index_pairs[index].second;

            const auto & chr1 = this->current[pop_base]->
                    population[this->current[pop_base]->fitness[pos1].second];

            const auto & chr2 = this->current[pop_guide]->
                    population[this->current[pop_base]->fitness[pos2].second];

            if(dist->distance(chr1, chr2) >= minimum_distance - 1e-6) {
                copy(begin(chr1), end(chr1), begin(initial_solution));
                copy(begin(chr2), end(chr2), begin(guiding_solution));
                found_pair = true;
                break;
            }

            index_pairs.erase(begin(index_pairs) + index);
            ++tested_pairs_count;
            elapsed_seconds = std::chrono::duration_cast<std::chrono::seconds>
                (std::chrono::system_clock::now() -
                 this->pr_start_time).count();
        }

        // The elite sets are too homogeneous, we cannot do
        // a good path relinking. Let's try other populations.
        if(!found_pair) {
            continue;
        }

        // Create a empty solution.
        std::pair<std::vector<double>, Chromosome> best_found;
        best_found.second.resize(this->current[0]->getChromosomeSize(), 0.0);

        best_found.first = std::vector<double>(this->OPT_SENSES.size());
        for(unsigned m = 0; m < this->OPT_SENSES.size(); m++) {
            if(this->OPT_SENSES[m] == Sense::MAXIMIZE) {
                best_found.first[0] = std::numeric_limits<double>::lowest();
            } else {
                best_found.first[0] = std::numeric_limits<double>::max();
            }
        }

        const auto fence = best_found.first;

        bool incumbentUpdated;

        // Perform the path relinking.
        if(pr_type == PathRelinking::Type::DIRECT) {
            incumbentUpdated = this->directPathRelink(initial_solution, 
                                                      guiding_solution, 
                                                      dist,
                                                      best_found, 
                                                      block_size, 
                                                      max_time, 
                                                      percentage);
        } else {
            incumbentUpdated =
                this->permutationBasedPathRelink(initial_solution, 
                                                 guiding_solution, 
                                                 dist,
                                                 best_found, 
                                                 block_size, 
                                                 max_time,
                                                 percentage);
        }

        if(!this->betterThan(best_found.first, fence)) {
            final_status |= PR::NO_IMPROVEMENT;
            continue;
        }

        // Re-decode and apply local search if the decoder are able to do it.
        best_found.first = this->decoder.decode(best_found.second, true);

        if(this->updateIncumbentSolutions({best_found})) {
            incumbentUpdated = true;
        }

        if(incumbentUpdated) {
            final_status |= PR::BEST_IMPROVEMENT;
        }

        // Now, check if the best solution found is really good.
        // If it is no worse than the best elite solution, overwrite the worse
        // elite solution in the population.
        bool include_in_population =
            !this->betterThan(this->current[pop_base]->fitness[0].first,
                    best_found.first);

        // If not the best, but is no worse than the worst elite member, check
        // if the distance between this solution and all elite members
        // is at least minimum_distance.
        if(!include_in_population &&
                !this->betterThan(
                    this->current[pop_base]->fitness[num_elites - 1].first,
                    best_found.first)) {
            include_in_population = true;
            for(unsigned i = 0; i < num_elites; ++i) {
                if(dist->distance(best_found.second,
                            this->current[pop_base]->
                            population[this->current[pop_base]->
                            fitness[i].second])
                        < minimum_distance - 1e-6) {
                    include_in_population = false;
                    final_status |= PR::NO_IMPROVEMENT;
                    break;
                }
            }
        }

        if(include_in_population) {
            std::copy(begin(best_found.second), end(best_found.second),
                      begin(this->current[pop_base]->
                                population[this->current[pop_base]->
                                    fitness.back().second]));

            this->current[pop_base]->fitness.back().first = best_found.first;
            // Reorder the chromosomes.
            this->current[pop_base]->sortFitness(this->OPT_SENSES);
            final_status |= PR::ELITE_IMPROVEMENT;
        }
    }

    return final_status;
}

//----------------------------------------------------------------------------//

template<class Decoder>
PathRelinking::PathRelinkingResult NSMPBRKGA<Decoder>::pathRelink(
        std::shared_ptr<DistanceFunctionBase> dist,
        long max_time) {

    size_t block_size = ceil(this->params.alpha_block_size *
                             sqrt(this->params.population_size));
    if(block_size > this->CHROMOSOME_SIZE) {
        block_size = this->CHROMOSOME_SIZE / 2;
    }

    return this->pathRelink(dist, 
                            this->params.pr_number_pairs, 
                            this->params.pr_minimum_distance,
                            this->params.pr_type, 
                            this->params.pr_selection, 
                            block_size, 
                            max_time,
                            this->params.pr_percentage);
}

//----------------------------------------------------------------------------//

// This is a multi-thread version. For small chromosomes, it may be slower than
// single thread version.
template<class Decoder>
bool NSMPBRKGA<Decoder>::directPathRelink(
        const Chromosome & chr1, 
        const Chromosome & chr2,
        std::shared_ptr<DistanceFunctionBase> dist,
        std::pair<std::vector<double>, Chromosome> & best_found,
        std::size_t block_size, 
        long max_time, 
        double percentage) {

    const std::size_t NUM_BLOCKS =
            std::size_t(ceil((double)chr1.size() / block_size));
    const std::size_t PATH_SIZE = std::size_t(percentage * NUM_BLOCKS);

    // Create the set of indices to test.
    std::set<std::size_t> remaining_blocks;
    for(std::size_t i = 0; i < NUM_BLOCKS; ++i) {
        remaining_blocks.insert(i);
    }
    Chromosome old_keys(chr1.size());

    struct Triple {
    public:
        Chromosome chr;
        std::vector<double> fitness;
        std::size_t block_index;
        Triple(): chr(), fitness(0), block_index(0) {}
    };

    // Allocate memory for the candidates.
    std::vector<Triple> candidates_left(NUM_BLOCKS);
    std::vector<Triple> candidates_right(NUM_BLOCKS);

    for(unsigned i = 0; i < candidates_left.size(); i++) {
        candidates_left[i].chr.resize(chr1.size());
    }

    for(unsigned i = 0; i < candidates_right.size(); i++) {
        candidates_right[i].chr.resize(chr1.size());
    }

    const Chromosome * base = & chr1;
    const Chromosome * guide = & chr2;
    std::vector<Triple> * candidates_base = & candidates_left;
    std::vector<Triple> * candidates_guide = & candidates_right;

    #ifdef _OPENMP
    #pragma omp parallel for num_threads(MAX_THREADS)
    #endif
    for(std::size_t i = 0; i < candidates_left.size(); ++i) {
        std::copy(begin(*base), end(*base), begin(candidates_left[i].chr));
    }

    #ifdef _OPENMP
    #pragma omp parallel for num_threads(MAX_THREADS)
    #endif
    for(std::size_t i = 0; i < candidates_right.size(); ++i) {
        std::copy(begin(*guide), end(*guide), begin(candidates_right[i].chr));
    }

    std::vector<std::pair<std::vector<double>, Chromosome>> newSolutions;

    std::size_t iterations = 0;
    while(!remaining_blocks.empty()) {
        // Set the block of keys from the guide solution for each candidate.
        auto it_block_idx = remaining_blocks.begin();

        for(std::size_t i = 0; i < remaining_blocks.size(); ++i) {
            const auto block_base = (*it_block_idx) * block_size;

            const auto it_key_block1 =
                    (*candidates_base)[i].chr.begin() + block_base;
            const auto it_key_block2 = guide->begin() + block_base;

            const auto bs = (block_base + block_size > guide->size())?
                            guide->size() - block_base : block_size;

            // If these keys do not affect the solution, skip them.
            if(!dist->affectSolution(it_key_block1, it_key_block2, bs)) {
                it_block_idx = remaining_blocks.erase(it_block_idx);
                --i;
                continue;
            }

            // Save the former keys before...
            std::copy_n((*candidates_base)[i].chr.begin() + block_base, bs,
                        old_keys.begin() + block_base);

            // ... copy the keys from the guide solution.
            std::copy_n(guide->begin() + block_base, bs,
                        (*candidates_base)[i].chr.begin() + block_base);

            (*candidates_base)[i].block_index = *it_block_idx;
            ++it_block_idx;
        }

        // Decode the candidates.
        volatile bool times_up = false;
        #ifdef _OPENMP
            #pragma omp parallel for num_threads(MAX_THREADS) shared(times_up) \
                schedule(static, 1)
        #endif
        for(std::size_t i = 0; i < remaining_blocks.size(); ++i) {
            (*candidates_base)[i].fitness =
                std::vector<double>(this->OPT_SENSES.size());
            for(unsigned m = 0; m < this->OPT_SENSES.size(); m++) {
                if(this->OPT_SENSES[m] == Sense::MAXIMIZE) {
                    (*candidates_base)[i].fitness[m] =
                        std::numeric_limits<double>::lowest();
                } else {
                    (*candidates_base)[i].fitness[m] =
                        std::numeric_limits<double>::max();
                }
            }

            if(times_up) {
                continue;
            }

            (*candidates_base)[i].fitness =
                    this->decoder.decode((*candidates_base)[i].chr, false);

            const auto elapsed_seconds =
                std::chrono::duration_cast<std::chrono::seconds>
                (std::chrono::system_clock::now() -
                 this->pr_start_time).count();
            if(elapsed_seconds > max_time) {
                times_up = true;
            }
        }

        std::vector<std::pair<std::vector<double>, Chromosome>>
            candidateSolutions(remaining_blocks.size());
        std::transform((*candidates_base).begin(), 
                       (*candidates_base).begin() + remaining_blocks.size(), 
                       candidateSolutions.begin(),
                       [](Triple candidate) {
                            return std::make_pair(candidate.fitness, 
                                                  candidate.chr);
                       });
        newSolutions.insert(newSolutions.end(), 
                            candidateSolutions.begin(),
                            candidateSolutions.end());

        // Locate the best candidate.
        std::size_t best_index = 0;
        std::size_t best_block_index = 0;

        std::vector<double> best_value(this->OPT_SENSES.size());
        for(unsigned m = 0; m < this->OPT_SENSES.size(); m++) {
            if(this->OPT_SENSES[m] == Sense::MAXIMIZE) {
                best_value[m] = std::numeric_limits<double>::lowest();
            } else {
                best_value[m] = std::numeric_limits<double>::max();
            }
        }

        for(std::size_t i = 0; i < remaining_blocks.size(); ++i) {
            if(this->betterThan((*candidates_base)[i].fitness, best_value)) {
                best_block_index = (*candidates_base)[i].block_index;
                best_value = (*candidates_base)[i].fitness;
                best_index = i;
            }
        }

        // Hold it, if it is the best found until now.
        if(this->betterThan((*candidates_base)[best_index].fitness, 
                            best_found.first)) {
            best_found.first = (*candidates_base)[best_index].fitness;
            std::copy(begin((*candidates_base)[best_index].chr),
                      end((*candidates_base)[best_index].chr),
                      begin(best_found.second));
        }

        // Restore original keys and copy the block of keys for all future
        // candidates. The last candidate will not be used.
        it_block_idx = remaining_blocks.begin();
        for(std::size_t i = 0; i < remaining_blocks.size() - 1;
            ++i, ++it_block_idx) {

            auto block_base = (*it_block_idx) * block_size;
            auto bs = (block_base + block_size > guide->size())?
                      guide->size() - block_base : block_size;

            std::copy_n(old_keys.begin() + block_base, bs,
                        (*candidates_base)[i].chr.begin() + block_base);

            // Recompute the offset for the best block.
            block_base = best_block_index * block_size;
            bs = (block_base + block_size > guide->size())?
                 guide->size() - block_base : block_size;

            std::copy_n((*candidates_base)[best_index].chr.begin() + block_base,
                        bs, (*candidates_base)[i].chr.begin() + block_base);
        }

        std::swap(base, guide);
        std::swap(candidates_base, candidates_guide);
        remaining_blocks.erase(best_block_index);

        const auto elapsed_seconds =
            std::chrono::duration_cast<std::chrono::seconds>
            (std::chrono::system_clock::now() - this->pr_start_time).count();

        if((elapsed_seconds > max_time) || (iterations++ > PATH_SIZE)) {
            break;
        }
    } // end while

    return this->updateIncumbentSolutions(newSolutions);
}

//----------------------------------------------------------------------------//

template<class Decoder>
bool NSMPBRKGA<Decoder>::permutationBasedPathRelink(
        Chromosome & chr1, 
        Chromosome & chr2,
        std::shared_ptr<DistanceFunctionBase> /*non-used*/,
        std::pair<std::vector<double>, Chromosome> & best_found,
        std::size_t /*non-used block_size*/,
        long max_time, 
        double percentage) {

    const std::size_t PATH_SIZE = std::size_t(percentage *
                                              this->CHROMOSOME_SIZE);

    std::set<std::size_t> remaining_indices;
    for(std::size_t i = 0; i < chr1.size(); ++i) {
        remaining_indices.insert(i);
    }

    struct DecodeStruct {
    public:
        Chromosome chr;
        std::vector<double> fitness;
        std::size_t key_index;
        std::size_t pos1;
        std::size_t pos2;
        DecodeStruct(): chr(), fitness(0),
                        key_index(0), pos1(0), pos2(0) {}
    };

    // Allocate memory for the candidates.
    std::vector<DecodeStruct> candidates_left(chr1.size());
    std::vector<DecodeStruct> candidates_right(chr1.size());

    for(unsigned i = 0; i < candidates_left.size(); i++) {
        candidates_left[i].chr.resize(chr1.size());
    }

    for(unsigned i = 0; i < candidates_right.size(); i++) {
        candidates_right[i].chr.resize(chr1.size());
    }

    Chromosome * base = & chr1;
    Chromosome * guide = & chr2;
    std::vector<DecodeStruct> * candidates_base = & candidates_left;
    std::vector<DecodeStruct> * candidates_guide = & candidates_right;

    std::vector<std::size_t> chr1_indices(chr1.size());
    std::vector<std::size_t> chr2_indices(chr1.size());
    std::vector<std::size_t> * base_indices = & chr1_indices;
    std::vector<std::size_t> * guide_indices = & chr2_indices;

    // Create and order the indices.
    std::vector<std::pair<std::vector<double>, std::size_t>>
        sorted(chr1.size());

    for(unsigned j = 0; j < 2; ++j) {
        for(std::size_t i = 0; i < base->size(); ++i) {
            sorted[i] = 
                std::pair<std::vector<double>, std::size_t>((*base)[i], i);
        }

        std::sort(begin(sorted), end(sorted));
        for(std::size_t i = 0; i < base->size(); ++i) {
            (*base_indices)[i] = sorted[i].second;
        }

        swap(base, guide);
        swap(base_indices, guide_indices);
    }

    base = & chr1;
    guide = & chr2;
    base_indices = & chr1_indices;
    guide_indices = & chr2_indices;

    #ifdef _OPENMP
    #pragma omp parallel for num_threads(MAX_THREADS)
    #endif
    for(std::size_t i = 0; i < candidates_left.size(); ++i) {
        std::copy(begin(*base), end(*base), begin(candidates_left[i].chr));
    }

    #ifdef _OPENMP
    #pragma omp parallel for num_threads(MAX_THREADS)
    #endif
    for(std::size_t i = 0; i < candidates_right.size(); ++i) {
        std::copy(begin(*guide), end(*guide), begin(candidates_right[i].chr));
    }

    std::vector<std::pair<std::vector<double>, Chromosome>> newSolutions;

    std::size_t iterations = 0;
    while(!remaining_indices.empty()) {
        std::size_t position_in_base;
        std::size_t position_in_guide;

        auto it_idx = remaining_indices.begin();
        for(std::size_t i = 0; i < remaining_indices.size(); ++i) {
            position_in_base = (*base_indices)[*it_idx];
            position_in_guide = (*guide_indices)[*it_idx];

            if(position_in_base == position_in_guide) {
                it_idx = remaining_indices.erase(it_idx);
                --i;
                continue;
            }

            (*candidates_base)[i].key_index = *it_idx;
            (*candidates_base)[i].pos1 = position_in_base;
            (*candidates_base)[i].pos2 = position_in_guide;
            (*candidates_base)[i].fitness =
                std::vector<double>(this->OPT_SENSES.size());
            for(unsigned m = 0; m < this->OPT_SENSES.size(); m++) {
                if(this->OPT_SENSES[m] == Sense::MAXIMIZE) {
                    (*candidates_base)[i].fitness[m] =
                        std::numeric_limits<double>::lowest();
                } else {
                    (*candidates_base)[i].fitness[m] =
                        std::numeric_limits<double>::max();
                }
            }

            ++it_idx;
        }

        if(remaining_indices.size() == 0) {
            break;
        }

        // Decode the candidates.
        volatile bool times_up = false;
        #ifdef _OPENMP
            #pragma omp parallel for num_threads(MAX_THREADS) shared(times_up) \
                schedule(static, 1)
        #endif
        for(std::size_t i = 0; i < remaining_indices.size(); ++i) {
            if(times_up) {
                continue;
            }

            std::swap((*candidates_base)[i].chr[(*candidates_base)[i].pos1],
                      (*candidates_base)[i].chr[(*candidates_base)[i].pos2]);

            (*candidates_base)[i].fitness =
                    this->decoder.decode((*candidates_base)[i].chr, false);

            std::swap((*candidates_base)[i].chr[(*candidates_base)[i].pos1],
                      (*candidates_base)[i].chr[(*candidates_base)[i].pos2]);

            const auto elapsed_seconds =
                std::chrono::duration_cast<std::chrono::seconds>
                (std::chrono::system_clock::now() -
                 this->pr_start_time).count();
            if(elapsed_seconds > max_time) {
                times_up = true;
            }
        }

        std::vector<std::pair<std::vector<double>, Chromosome>>
            candidateSolutions(remaining_indices.size());
        std::transform((*candidates_base).begin(), 
                       (*candidates_base).begin() + remaining_indices.size(), 
                       candidateSolutions.begin(),
                       [](DecodeStruct candidate) {
                            return std::make_pair(candidate.fitness, 
                                                  candidate.chr);
                       });
        newSolutions.insert(newSolutions.end(), 
                            candidateSolutions.begin(),
                            candidateSolutions.end());

        // Locate the best candidate
        std::size_t best_key_index = 0;

        std::size_t best_index;
        std::vector<double> best_value(this->OPT_SENSES.size());
        for(unsigned m = 0; m < this->OPT_SENSES.size(); m++) {
            if(this->OPT_SENSES[m] == Sense::MAXIMIZE) {
                best_value[m] = std::numeric_limits<double>::lowest();
            } else {
                best_value[m] = std::numeric_limits<double>::max();
            }
        }

        for(std::size_t i = 0; i < remaining_indices.size(); ++i) {
            if(this->betterThan((*candidates_base)[i].fitness, best_value)) {
                best_index = i;
                best_key_index = (*candidates_base)[i].key_index;
                best_value = (*candidates_base)[i].fitness;
            }
        }

        position_in_base = (*base_indices)[best_key_index];
        position_in_guide = (*guide_indices)[best_key_index];

        // Commit the best exchange in all candidates.
        // The last will not be used.
        for(std::size_t i = 0; i < remaining_indices.size() - 1; ++i) {
            std::swap((*candidates_base)[i].chr[position_in_base],
                      (*candidates_base)[i].chr[position_in_guide]);
        }

        std::swap((*base_indices)[position_in_base],
                  (*base_indices)[position_in_guide]);

        // Hold, if it is the best found until now
        if(this->betterThan(best_value, best_found.first)) {
            const auto & best_chr = (*candidates_base)[best_index].chr;
            best_found.first = best_value;
            copy(begin(best_chr), end(best_chr), begin(best_found.second));
        }

        std::swap(base_indices, guide_indices);
        std::swap(candidates_base, candidates_guide);
        remaining_indices.erase(best_key_index);

        // Is time to stop?
        const auto elapsed_seconds =
            std::chrono::duration_cast<std::chrono::seconds>
            (std::chrono::system_clock::now() - this->pr_start_time).count();

        if((elapsed_seconds > max_time) || (iterations++ > PATH_SIZE)) {
            break;
        }
    }

    return this->updateIncumbentSolutions(newSolutions);
}

//----------------------------------------------------------------------------//

template<class Decoder>
bool NSMPBRKGA<Decoder>::updateIncumbentSolutions(
        std::vector<std::pair<std::vector<double>, Chromosome>> newSolutions) {
    bool result = false;

    if(newSolutions.empty()) {
        return result;
    }

    newSolutions = 
        Population::nonDominatedSort<Chromosome>(newSolutions,
                                                 this->OPT_SENSES).front();

    for(unsigned i = 0; i < newSolutions.size(); i++) {
        bool isDominatedOrEqual = false;

        for(auto it = this->incumbentSolutions.begin(); 
                it != this->incumbentSolutions.end();) {
            auto incumbentSolution = *it;

            if(Population::betterThan(newSolutions[i].first, 
                                      incumbentSolution.first, 
                                      this->OPT_SENSES)) {
                it = this->incumbentSolutions.erase(it);
            } else {
                if(Population::betterThan(incumbentSolution.first, 
                                          newSolutions[i].first, 
                                          this->OPT_SENSES) ||
                        std::equal(incumbentSolution.first.begin(),
                                   incumbentSolution.first.end(),
                                   newSolutions[i].first.begin(), 
                                   [](double a, double b) {
                                        return fabs(a - b) < 
                                        std::numeric_limits<double>::epsilon();
                                   })) {
                    isDominatedOrEqual = true;
                    break;
                }

                it++;
            }
        }

        if(!isDominatedOrEqual) {
            this->incumbentSolutions.push_back(newSolutions[i]);
            result = true;
        }
    }

    if(this->params.num_incumbent_solutions > 0 &&
            this->incumbentSolutions.size() >
            this->params.num_incumbent_solutions) {
        Population::crowdingSort<Chromosome>(this->incumbentSolutions);
        this->incumbentSolutions.resize(this->params.num_incumbent_solutions);
        result = true;
    }

    return result;
}

//----------------------------------------------------------------------------//

template<class Decoder>
inline double NSMPBRKGA<Decoder>::rand01() {
    // **NOTE:** instead to use std::generate_canonical<> (which can be
    // a little bit slow), we may use
    //    rng() * (1.0 / std::numeric_limits<std::mt19937::result_type>::max());
    // However, this approach has some precision problems on some platforms
    // (notably Linux)

    return std::generate_canonical<double, std::numeric_limits<double>::digits>
          (this->rng);
}

//----------------------------------------------------------------------------//

template<class Decoder>
inline uint_fast32_t NSMPBRKGA<Decoder>::randInt(const uint_fast32_t n) {
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
        i = this->rng() & used;  // Toss unused bits to shorten search.
    } while(i > n);
    return i;
}

} // end namespace NSMPBRKGA

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

/// Template specialization to BRKGA::Sense.
template <>
INLINE const std::vector<std::string> &
EnumIO<BRKGA::Sense>::enum_names() {
    static std::vector<std::string> enum_names_({
        "MINIMIZE",
        "MAXIMIZE"
    });
    return enum_names_;
}

/// Template specialization to BRKGA::PathRelinking::Type.
template <>
INLINE const std::vector<std::string> &
EnumIO<BRKGA::PathRelinking::Type>::enum_names() {
    static std::vector<std::string> enum_names_({
        "DIRECT",
        "PERMUTATION"
    });
    return enum_names_;
}

/// Template specialization to BRKGA::PathRelinking::Selection.
template <>
INLINE const std::vector<std::string> &
EnumIO<BRKGA::PathRelinking::Selection>::enum_names() {
    static std::vector<std::string> enum_names_({
        "BESTSOLUTION",
        "RANDOMELITE"
    });
    return enum_names_;
}

/// Template specialization to BRKGA::BiasFunctionType.
template <>
INLINE const std::vector<std::string> &
EnumIO<BRKGA::BiasFunctionType>::enum_names() {
    static std::vector<std::string> enum_names_({
        "CONSTANT",
        "CUBIC",
        "EXPONENTIAL",
        "LINEAR",
        "LOGINVERSE",
        "QUADRATIC",
        "CUSTOM"
    });
    return enum_names_;
}
///@}

#endif // NSMPBRKGA_HPP_

