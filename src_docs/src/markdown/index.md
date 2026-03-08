NS-BRKGA Guide and Documentation {#mainpage}
================================================================================

NS-BRKGA provides a _very easy-to-use_ framework for the
**Non-dominated Sorting Biased Random-Key Genetic Algorithm (NS-BRKGA)**,
a multi-objective evolutionary metaheuristic that combines ideas from
Non-dominated Sorting (as in NSGA-II) with the Biased Random-Key Genetic
Algorithm with Multiple Parents and Implicit Path Relink (BRKGA-MP-IPR).
Assuming you have a _decoder_ for your problem, you can set up, run, and
extract the Pareto front in just a few commands.

This C++ implementation provides a fast-prototyping API using **C++17**
standards and libraries. All code was developed as a **header-only library**
and has no external dependencies other than the standard library and the
bundled utilities. Just copy/check out the files and point your compiler's
header path to the `nsbrkga/` folder (`-I` flag for G++ and Clang++).

The framework leverages multi-threading for decoding by setting a single
parameter (assuming your decoder is thread-safe). The parallelism follows
the same philosophy as BRKGA-MP-IPR: independent sub-populations that
periodically share elite solutions.

The code compiles with [GCC ≥ 7.2](https://gcc.gnu.org) and
[Clang ≥ 6.0](https://clang.llvm.org), and is very likely to compile with
other modern C++ compilers. Multi-threading requires
[OpenMP](https://www.openmp.org).

If you are not familiar with BRKGA, we recommend reading:

- [Standard BRKGA](http://dx.doi.org/10.1007/s10732-010-9143-1): Gonçalves &
  Resende (2011).
- [BRKGA-MP-IPR](http://dx.doi.org/10.1016/j.ejor.2021.10.047): Andrade
  et al. (2021).
- [NSGA-II](https://doi.org/10.1109/4235.996017): Deb et al. (2002).

If you already know what _elite set_, _decoder_, and _non-dominated front_
mean, head straight to the [Guide](@ref guide).

Repository
--------------------------------------------------------------------------------

The source code is available at:

> <https://github.com/luishpmendes/ns-brkga>

License and Citing
--------------------------------------------------------------------------------

NS-BRKGA is distributed under a permissive BSD-like license and may be used
freely. Since this framework is also part of an academic effort, we kindly
ask you to cite the originating work in any publication that uses or
references this software.

> L.H.P. Mendes. *NS-BRKGA: Non-dominated Sorting Biased Random-Key Genetic
> Algorithm*. Source code available at:
> <https://github.com/luishpmendes/ns-brkga>.

About the logo
--------------------------------------------------------------------------------

The logo represents the interplay between non-dominated sorting fronts and the
chromosome crossover process at the heart of BRKGA — multiple solution paths
merging and diverging as the population evolves toward the Pareto front.
