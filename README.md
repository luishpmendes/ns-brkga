<div align="center">
  <img src="https://github.com/luishpmendes/ns-brkga/blob/master/src_docs/src/assets/logo_name_300.png">
</div>

# NS-BRKGA

**NS-BRKGA** (Non-dominated Sorting Biased Random-Key Genetic Algorithm) is a header-only C++ framework for **multi-objective optimization** based on **biased random keys**, **non-dominated sorting**, and **implicit path relinking**.

This repository started as a fork of **BRKGA-MP-IPR**, but it has evolved toward a different goal: instead of relying on lexicographic multi-objective handling, **NS-BRKGA maintains a Pareto-style non-dominated set of incumbent solutions**, ranks solutions by fronts, and uses crowding-based diversity preservation.

## Why this project exists

The original BRKGA-MP-IPR is an excellent framework for biased random-key genetic algorithms with multi-parent crossover and implicit path relinking. NS-BRKGA builds on that foundation and adapts it to scenarios where the objective is not a single best solution or a lexicographic priority order, but rather a **set of well-distributed non-dominated solutions**.

In practical terms, NS-BRKGA is designed for problems in which:

- the decoder returns **multiple objective values**;
- the optimization sense may differ by objective;
- you want to preserve a **Pareto front approximation** instead of a single incumbent;
- chromosome rewriting during decoding is useful, enabling **Lamarckian local improvement**;
- diversity matters both in the population and in the stored incumbent solutions.

## Main features

- Header-only C++ implementation.
- Multi-objective optimization using **non-dominated sorting**.
- **Crowding-based ordering** inside Pareto fronts.
- **Multi-population** / island-model search.
- **Multi-parent** mating with configurable bias functions.
- Configurable crossover with:
  - `ROULETTE`
  - `GEOMETRIC`
- Optional **Lamarckian write-back** in the decoder through `decode(chromosome, rewrite)`.
- Optional **implicit path relinking** with multiple strategies:
  - `ALLOCATION`
  - `PERMUTATION`
  - `BINARY_SEARCH`
- Built-in support for:
  - elite exchange between populations,
  - shaking,
  - resets,
  - warm starts,
  - chromosome injection,
  - custom bias functions,
  - custom diversity functions,
  - custom distance functions.
- Parallel decoding with OpenMP when the decoder is thread-safe.

## Repository layout

```text
ns-brkga/
├── nsbrkga/        # library headers
├── examples/       # usage examples
├── test/           # tests
├── src_docs/       # documentation sources
└── docs/           # generated documentation assets
```

## Installation

NS-BRKGA is a **header-only** library. Clone the repository and add its root folder to your compiler include path.

```bash
git clone https://github.com/luishpmendes/ns-brkga.git
```

A minimal compilation command is:

```bash
g++ -std=c++14 -O3 -fopenmp -I/path/to/ns-brkga your_program.cpp -o your_program
```

Then include the main header in your code:

```cpp
#include "nsbrkga/nsbrkga.hpp"
```

## Decoder interface

Your decoder must expose the following interface:

```cpp
std::vector<double> decode(NSBRKGA::Chromosome& chromosome, bool rewrite);
```

The return value is the objective vector.

- `chromosome` is a random-key vector with alleles in `[0, 1)`.
- `rewrite == true` means the decoder may rewrite the chromosome.
- `rewrite == false` means the decoder must preserve the chromosome.
- If you use more than one thread, the decoder must be **thread-safe**.

## Minimal example

```cpp
#include "nsbrkga/nsbrkga.hpp"

#include <iostream>
#include <vector>

struct MyDecoder {
    std::vector<double> decode(NSBRKGA::Chromosome& chromosome,
                               bool /* rewrite */) const {
        double obj1 = 0.0;
        double obj2 = 0.0;

        for(std::size_t i = 0; i < chromosome.size(); ++i) {
            obj1 += chromosome[i];
            obj2 += (i + 1) * chromosome[i];
        }

        return {obj1, obj2};
    }
};

int main() {
    auto [params, control] = NSBRKGA::readConfiguration("nsbrkga.conf");

    MyDecoder decoder;

    NSBRKGA::NSBRKGA<MyDecoder> algorithm(
        decoder,
        {NSBRKGA::Sense::MINIMIZE, NSBRKGA::Sense::MINIMIZE},
        42,     // seed
        100,    // chromosome size
        params,
        1       // max threads
    );

    algorithm.initialize();

    for(unsigned generation = 1; generation <= 200; ++generation) {
        algorithm.evolve();

        if(control.exchange_interval > 0 &&
           generation % control.exchange_interval == 0) {
            algorithm.exchangeElite(control.num_exchange_individuals);
        }
    }

    const auto& incumbents = algorithm.getIncumbentSolutions();
    for(const auto& solution : incumbents) {
        const auto& fitness = solution.first;
        std::cout << "(" << fitness[0] << ", " << fitness[1] << ")\n";
    }

    return 0;
}
```

## Typical workflow

1. Build a problem data structure.
2. Implement a decoder that maps a chromosome to a vector of objective values.
3. Define the optimization senses, one per objective.
4. Load parameters with `NSBRKGA::readConfiguration()` or build them manually.
5. Instantiate `NSBRKGA::NSBRKGA<Decoder>`.
6. Optionally inject warm-start solutions.
7. Call `initialize()`.
8. Run `evolve()` for one or more generations.
9. Optionally call `exchangeElite()`, `pathRelink()`, `shake()`, or `reset()`.
10. Retrieve the incumbent non-dominated solutions with `getIncumbentSolutions()`.

## Configuration file

The helper `NSBRKGA::readConfiguration()` reads key-value pairs from a text file. A complete example is shown below.

```text
POPULATION_SIZE 200
MIN_ELITES_PERCENTAGE 0.10
MAX_ELITES_PERCENTAGE 0.30
MUTATION_PROBABILITY 0.05
MUTATION_DISTRIBUTION 20
NUM_ELITE_PARENTS 2
TOTAL_PARENTS 3
BIAS_TYPE LOGINVERSE
DIVERSITY_TYPE AVERAGE_DISTANCE_BETWEEN_ALL_PAIRS
CROSSOVER_TYPE ROULETTE
NUM_INDEPENDENT_POPULATIONS 3
NUM_INCUMBENT_SOLUTIONS 0
PR_TYPE PERMUTATION
PR_PERCENTAGE 0.20
EXCHANGE_INTERVAL 50
NUM_EXCHANGE_INDIVIDUALS 2
PATH_RELINK_INTERVAL 100
SHAKE_INTERVAL 0
RESET_INTERVAL 0
```

Notes:

- Keys are case-insensitive.
- Blank lines and lines starting with `#` are ignored.
- `CROSSOVER_TYPE` is optional for backward compatibility.
- A value of `0` for exchange, relinking, shaking, or reset intervals disables that operation in the outer control loop.

## Parameter highlights

### Bias function presets

- `CONSTANT`
- `LINEAR`
- `QUADRATIC`
- `CUBIC`
- `LOGINVERSE`
- `SQRT`
- `CBRT`
- `EXPONENTIAL`
- `CUSTOM`

### Diversity function presets

- `NONE`
- `AVERAGE_DISTANCE_TO_CENTROID`
- `AVERAGE_DISTANCE_BETWEEN_ALL_PAIRS`
- `POWER_MEAN_BASED`
- `CUSTOM`

### Distance functions for path relinking

The framework provides built-in distance functions such as:

- Hamming distance,
- Kendall tau distance,
- Euclidean distance,
- or a user-defined custom distance function.

## Important API notes

- The constructor receives a **vector of senses**, not a single sense.
- The decoder returns a **`std::vector<double>`**, not a scalar fitness.
- The main result is a **set of incumbent non-dominated solutions**, not only one best chromosome.
- Within a front, NS-BRKGA uses **crowding-based ordering** to preserve spread.
- Path relinking evaluates intermediate candidates with `rewrite = false` and only rewrites the final chosen solution.

## Relationship to BRKGA-MP-IPR

NS-BRKGA is not just a rename of the original project.

It keeps the random-key, multi-parent, and path-relinking spirit of BRKGA-MP-IPR, but changes the optimization logic to better support **Pareto-based multi-objective search**. In particular, the current codebase introduces or emphasizes:

- non-dominated sorting instead of lexicographic ordering,
- crowding-based ranking inside fronts,
- storage of multiple incumbent solutions,
- diversity-aware elite sizing,
- explicit support for decoder rewrite in a multi-objective setting,
- a configurable crossover operator with discrete or geometric recombination.

## References

If this project contributes to academic work, please consider citing the foundational papers behind the framework lineage:

- José Fernando Gonçalves and Mauricio G. C. Resende.
  **Biased random-key genetic algorithms for combinatorial optimization**.
  *Journal of Heuristics*, 17:487-525, 2011.
  DOI: [10.1007/s10732-010-9143-1](https://doi.org/10.1007/s10732-010-9143-1)

- Carlos E. Andrade, Rodrigo F. Toso, José F. Gonçalves, and Mauricio G. C. Resende.
  **The Multi-Parent Biased Random-Key Genetic Algorithm with Implicit Path-Relinking and its real-world applications**.
  *European Journal of Operational Research*, 289(1):17-30, 2021.
  DOI: [10.1016/j.ejor.2019.11.037](https://doi.org/10.1016/j.ejor.2019.11.037)

If you use or report specifically on the multi-objective lineage inherited from the upstream codebase, you may also find the following reference relevant:

- Carlos E. Andrade, Leonardo S. Pessoa, and Sebastian Stawiarski.
  **The Physical Cell Identity Assignment Problem: a Multi-objective Optimization Approach**.
  *IEEE Transactions on Evolutionary Computation*, 27(1):130-144, 2023.
  DOI: [10.1109/TEVC.2022.3185927](https://doi.org/10.1109/TEVC.2022.3185927)

## License

This project is distributed under the terms described in [LICENSE.md](LICENSE.md).

## Contributing

Please see [CONTRIBUTING.md](CONTRIBUTING.md).
