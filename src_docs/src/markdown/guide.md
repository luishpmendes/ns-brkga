Guide / Tutorial {#guide}
===============

[TOC]

Installation {#guide_installation}
================================================================================

NS-BRKGA is a header-only framework: download the headers and point your
compiler's include path to the `nsbrkga/` folder.

Quick example (unix): clone the repository first:

    $ git clone https://github.com/luishpmendes/ns-brkga
    Cloning into 'ns-brkga'...

Let's write a `test.cpp` to verify the library is found:

```cpp
#include "nsbrkga.hpp"
#include <iostream>

int main() {
    std::cout << "NS-BRKGA loaded. Sense: "
              << NSBRKGA::Sense::MINIMIZE << "\n";
    return 0;
}
```

Compile and run:

    $ g++ -std=c++17 -Ins-brkga/nsbrkga test.cpp -o test
    $ ./test
    NS-BRKGA loaded. Sense: MINIMIZE

To build this documentation you also need
[Doxygen](http://www.doxygen.nl), [Doxyrest](https://github.com/vovkos/doxyrest),
and [Sphinx](https://www.sphinx-doc.org) with `sphinx-rtd-theme`.

TL;DR {#guide_tldr}
================================================================================

The fastest path is to look at the examples in the repository under
[`examples/`](https://github.com/luishpmendes/ns-brkga/tree/master/examples).

The basic usage pattern is:

```cpp
#include "nsbrkga.hpp"
#include <iostream>
#include <stdexcept>
#include <string>

int main(int argc, char* argv[]) {
    // 1. Load your instance data
    auto instance = MyInstance(argv[1]);

    // 2. Read BRKGA/NS-BRKGA parameters from a config file
    auto [brkga_params, control_params] =
        NSBRKGA::readConfiguration(argv[2]);

    // 3. Create the decoder
    MyDecoder decoder(instance);

    // 4. Build the NS-BRKGA object
    NSBRKGA::NSBRKGA<MyDecoder> algorithm(
        decoder,
        {NSBRKGA::Sense::MINIMIZE},   // one or more objectives
        /*seed=*/ 42,
        instance.num_elements,        // chromosome length
        brkga_params);

    // 5. Initialize populations
    algorithm.initialize();

    // 6. Evolve
    algorithm.evolve(/*num_generations=*/ 200);

    // 7. Retrieve the Pareto front
    auto front = algorithm.getCurrentFront();
    std::cout << "Pareto front size: " << front.size() << "\n";
    return 0;
}
```

Key steps at a glance:

1. Create an instance data structure and pass it to the decoder.
2. Implement a `decode()` method returning a vector of fitness values
   (one per objective).
3. Load parameters with `NSBRKGA::readConfiguration()`.
4. Build `NSBRKGA::BRKGA<Decoder>` specifying optimization senses and
   chromosome length.
5. Call `initialize()` and then `evolve()`.
6. Retrieve the Pareto front via `getCurrentFront()`.

Getting started {#guide_getting_started}
================================================================================

NS-BRKGA is an extension of BRKGA-MP-IPR for **multi-objective** problems.
Like its predecessor, it requires a _decoder_: a function that maps a
chromosome (a vector of real numbers in [0,1]) to one or more solution values.
Non-dominated sorting (NSGA-II style) replaces scalar fitness ranking so that
the algorithm naturally maintains and explores the Pareto front.

Before going further, explore the `examples/` folder in
[the repository](https://github.com/luishpmendes/ns-brkga). Each example
shows how to:
- Define a problem instance;
- Implement a multi-objective decoder;
- Set up and run NS-BRKGA;
- Retrieve and print the Pareto front.

First things first: the decoder function {#guide_decoder}
================================================================================

The decoder is the heart of NS-BRKGA.  It maps chromosomes to solutions.
The required interface is:

```cpp
class Decoder {
public:
    std::vector<double> decode(NSBRKGA::Chromosome& chromosome,
                               bool rewrite = true);
};
```

The `decode()` method must:
- Accept a `NSBRKGA::Chromosome&` and an optional `rewrite` flag.
- Return `std::vector<double>` — **one value per objective** (e.g., two
  values for a bi-objective problem).
- Be **thread-safe** when multi-threading is enabled.

## The chromosome

`NSBRKGA::Chromosome` is defined in `chromosome.hpp`:

```cpp
typedef std::vector<double> Chromosome;
```

Each gene is a `double` in [0,1]. Your decoder interprets the gene ordering
to construct a solution (e.g., sort by gene value to obtain a permutation).

You may subclass `std::vector<double>` to carry extra data:

```cpp
class Chromosome : public std::vector<double> {
public:
    double makespan;
    double total_completion_time;
    // ...
};
```

> **Warning:** Avoid polymorphism through base-class pointers — `std::vector`
> has no virtual destructor.

## Example: bi-objective decoder

```cpp
#include "nsbrkga.hpp"

class MyDecoder {
public:
    MyDecoder(const MyInstance& inst) : instance(inst) {}

    std::vector<double> decode(NSBRKGA::Chromosome& chr,
                               bool /* rewrite */) {
        // Build a permutation from chromosome keys
        std::vector<std::pair<double,unsigned>> perm(instance.n);
        for (unsigned i = 0; i < instance.n; ++i)
            perm[i] = {chr[i], i};
        std::sort(perm.begin(), perm.end());

        double obj1 = compute_objective1(perm, instance);
        double obj2 = compute_objective2(perm, instance);
        return {obj1, obj2};
    }

private:
    const MyInstance& instance;
};
```

Building the NS-BRKGA algorithm object {#guide_brkga_object}
================================================================================

`NSBRKGA::NSBRKGA` is the main class.  It is a template parameterized by the
decoder type to avoid virtual dispatch overhead.

Constructor signature:

```cpp
NSBRKGA(
    Decoder& decoder_reference,
    const std::vector<Sense>& senses,     // one per objective
    const unsigned seed,
    const unsigned chromosome_size,
    const NsbrkgaParams& params,
    const unsigned max_threads = 1,
    const bool evolutionary_mechanism_on = true);
```

- `senses`: `NSBRKGA::Sense::MINIMIZE` or `NSBRKGA::Sense::MAXIMIZE` for each
  objective.
- `chromosome_size`: length of each chromosome (problem-specific).
- `params`: read from a config file or built by hand.

Reading parameters:

```cpp
auto [brkga_params, control_params] =
    NSBRKGA::readConfiguration("config.conf");
```

A typical config file:

```
population_size 500
elite_percentage 0.30
mutants_percentage 0.15
num_elite_parents 2
total_parents 3
bias_type LOGINVERSE
num_independent_populations 3
exchange_interval 200
num_exchange_individuals 2
reset_interval 600
```

Initialization and warm-start solutions {#guide_algo_init}
================================================================================

Call `initialize()` before any other algorithm method:

```cpp
algorithm.initialize();
```

> **Warning:** `initialize()` must be called before `evolve()` or any other
> optimization method.

### Warm-start solutions

Encode good solutions from fast heuristics as chromosomes and inject them
before initialization:

```cpp
NSBRKGA::Chromosome warm(instance.n);
// fill warm[i] with keys in [0,1] encoding a known good solution
algorithm.setInitialPopulation({warm});
// Then:
algorithm.initialize();
```

Optimization: evolving the population {#guide_opt}
================================================================================

```cpp
algorithm.evolve(num_generations);
```

For fine-grained control, evolve one generation at a time inside a loop:

```cpp
for (unsigned gen = 0; gen < max_generations; ++gen) {
    algorithm.evolve(1);
    if (gen % exchange_interval == 0)
        algorithm.exchangeElite(num_exchange_individuals);
    if (time_limit_reached()) break;
}
```

Accessing the Pareto front {#guide_access_solutions}
================================================================================

After optimization, retrieve the non-dominated front:

```cpp
auto front = algorithm.getCurrentFront();
for (const auto& solution : front) {
    // solution.fitness is std::vector<double>
    // solution.chromosome is NSBRKGA::Chromosome
    std::cout << solution.fitness[0] << " " << solution.fitness[1] << "\n";
}
```

You can also inject modified chromosomes back into the population:

```cpp
algorithm.injectChromosome(improved_chromosome,
                           /*population_index=*/ 0,
                           /*position=*/ 0);
```

Multi-threading {#guide_tips_multihreading}
================================================================================

Set `max_threads` in the constructor to use OpenMP-based parallel decoding:

```cpp
NSBRKGA::NSBRKGA<MyDecoder> algorithm(
    decoder, senses, seed, chrom_size, params,
    /*max_threads=*/ std::thread::hardware_concurrency());
```

Your decoder **must be thread-safe** when `max_threads > 1`. The simplest
approach is to allocate all temporary buffers locally inside `decode()`.

Resetting and shaking the population {#guide_reset}
================================================================================

To escape local optima, reset or shake the population:

```cpp
// Full reset: re-initialize all populations from scratch
algorithm.reset();

// Shaking: perturb elite chromosomes while preserving diversity
algorithm.shake(num_shaken, NSBRKGA::ShakingType::CHANGE, population_index);
```

Troubleshooting {#guide_troubleshooting}
================================================================================

| Symptom | Likely cause | Fix |
|---------|-------------|-----|
| Pareto front size = 1 | All objectives agree / collide | Verify decoder returns physically distinct objective vectors |
| No improvement after many generations | Population too small or mutation too high | Increase `population_size`, reduce `mutants_percentage` |
| Compile error: `NSBRKGA` not found | Wrong include path | Ensure `-Ins-brkga/nsbrkga` is on the compiler flags |
| Segfault in decode | Decoder not thread-safe | Use local variables or per-thread storage inside `decode()` |

Further reading {#guide_further}
================================================================================

- Source: <https://github.com/luishpmendes/ns-brkga>
- BRKGA-MP-IPR (upstream): <https://github.com/ceandrade/brkga_mp_ipr_cpp>
- NSGA-II: Deb et al., *A fast and elitist multiobjective genetic algorithm:
  NSGA-II*, IEEE Trans. Evol. Comput. 6(2):182–197, 2002.
  DOI: [10.1109/4235.996017](https://doi.org/10.1109/4235.996017)
- BRKGA-MP-IPR: Andrade et al., *The multi-parent biased random-key genetic
  algorithm with implicit path-relinking*, Eur. J. Oper. Res. 289(2):545–557,
  2021. DOI: [10.1016/j.ejor.2021.10.047](https://doi.org/10.1016/j.ejor.2021.10.047)
