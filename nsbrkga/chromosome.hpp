/*******************************************************************************
 * chromosome.hpp: Interface for Chromosome class/structure.
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
 * \file chromosome.hpp
 * \brief Defines the Chromosome type used throughout the NS-BRKGA library.
 *
 * \details
 * A Chromosome is the fundamental unit of the genetic algorithm: a fixed-length
 * vector of double-precision alleles, each drawn from the continuous unit
 * interval \f$[0, 1]\f$.  The NS-BRKGA engine generates, stores, and
 * manipulates chromosomes; the user-supplied decoder is responsible for mapping
 * each chromosome to a feasible problem solution and returning the
 * corresponding objective values.
 *
 * **Allele invariant:** every allele \f$x_i\f$ satisfies
 * \f$0 \le x_i < 1\f$.  Values at the boundary 1.0 may arise from
 * floating-point arithmetic but are considered degenerate; the library clamps
 * generated alleles below 1.0 during mutation.
 *
 * **Chromosome length:** the length \f$n\f$ equals the problem dimension
 * (number of decision variables or encoding positions) and is fixed at
 * construction time via `NSBRKGA::NSBRKGA::NSBRKGA()`.
 *
 * **Decoder contract:**
 * - Input: a `const Chromosome &` (or a non-const reference when write-back
 *   is allowed).
 * - Output: `std::vector<double>` of objective values (one per optimization
 *   sense/objective).
 * - The decoder receives a `bool rewrite` flag.  When `rewrite == true` the
 *   decoder **may** overwrite the chromosome alleles (e.g., to store a locally
 *   improved encoding).  All modified allele values must remain in \f$[0,1)\f$.
 *   When `rewrite == false` the chromosome must not be modified.
 * - During path relinking the library always calls `decode()` with
 *   `rewrite = false` to preserve the path trajectory; it issues one final
 *   call with `rewrite = true` on the best chromosome found.
 *
 * **Thread safety:** when `NSBRKGA::NSBRKGA` is configured with more than one
 * thread, the library calls `decode()` for different chromosomes concurrently.
 * The decoder must be thread-safe (e.g., no unsynchronized access to shared
 * mutable state).  Per-thread working memory (local variables, thread-local
 * storage) is the recommended pattern.
 *
 * **Type alias rationale:** `Chromosome` is a plain `typedef` for
 * `std::vector<double>`.  Using a named alias makes the library's API more
 * self-documenting without imposing any overhead, and leaves open the option
 * of changing the underlying representation in the future without modifying
 * call sites.
 *
 * \see NSBRKGA::NSBRKGA for the algorithm class.
 * \see NSBRKGA::NsbrkgaParams for configuration parameters.
 * \see NSBRKGA::Population for population-level chromosome access.
 */

#ifndef NSBRKGA_CHROMOSOME_HPP_
#define NSBRKGA_CHROMOSOME_HPP_

#include <vector>

namespace NSBRKGA {

/**
 * \brief Chromosome representation: a vector of alleles in \f$[0,1)\f$.
 *
 * \details
 * `Chromosome` is a type alias for `std::vector<double>`.  Each element
 * (allele) represents a continuous random key in the unit interval
 * \f$[0, 1)\f$, so a chromosome of length \f$n\f$ defines a point in the
 * \f$n\f$-dimensional unit hypercube \f$[0,1)^n\f$.
 *
 * Double precision is used because single-precision floats may be insufficient
 * for problems that require fine-grained rank distinctions among alleles (e.g.,
 * permutation-based decoders that resolve ties by value).
 *
 * \warning Do **not** change the size of the chromosome inside the decoder;
 *          the library assumes `chromosome.size() == chromosome_size` at all
 *          times.
 *
 * \see NSBRKGA::NSBRKGA for parameter and usage details.
 * \see NSBRKGA::Population for sorted access to chromosomes in a population.
 */
typedef std::vector<double> Chromosome;

} // namespace NSBRKGA

#endif // NSBRKGA_CHROMOSOME_HPP_
