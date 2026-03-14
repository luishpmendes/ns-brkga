/**
 * Regression test for Population::nonDominatedSort.
 *
 * Exercises empty inputs, 1-objective, 2-objective, 3-objective, and
 * tie-handling cases and verifies front partitioning matches expected
 * results.
 *
 * Build:
 *   g++ -std=c++14 -I../nsbrkga test_nondominated_sort.cpp -o test_nds
 * Run:
 *   ./test_nds
 */

#include "nsbrkga.hpp"

#include <cassert>
#include <cstdlib>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

using Pair = std::pair<std::vector<double>, unsigned>;
using Front = std::vector<Pair>;
using Fronts = std::vector<Front>;

static int test_count = 0;
static int pass_count = 0;
static int fail_count = 0;

static void check(const std::string &name, bool condition) {
    test_count++;

    if (condition) {
        pass_count++;
        std::cout << "  PASS: " << name << std::endl;
    } else {
        fail_count++;
        std::cout << "  FAIL: " << name << std::endl;
    }
}

// Helper: extract only the fitness vectors from a front.
static std::vector<std::vector<double>> fitnessOf(const Front &f) {
    std::vector<std::vector<double>> r;
    r.reserve(f.size());

    for (auto &p : f) {
        r.push_back(p.first);
    }

    return r;
}

// ---- Test cases -----------------------------------------------------------

static void test_empty_fitness() {
    std::cout << "[test_empty_fitness]" << std::endl;
    std::vector<Pair> fitness;
    std::vector<NSBRKGA::Sense> senses = {NSBRKGA::Sense::MINIMIZE};
    auto fronts = NSBRKGA::Population::nonDominatedSort<unsigned>(fitness, senses);
    check("empty fitness → empty fronts", fronts.empty());
}

static void test_empty_senses() {
    std::cout << "[test_empty_senses]" << std::endl;
    std::vector<Pair> fitness = {{{1.0}, 0}};
    std::vector<NSBRKGA::Sense> senses;
    auto fronts = NSBRKGA::Population::nonDominatedSort<unsigned>(fitness, senses);
    check("empty senses → empty fronts", fronts.empty());
}

static void test_single_element() {
    std::cout << "[test_single_element]" << std::endl;
    std::vector<Pair> fitness = {{{3.0, 4.0}, 0}};
    std::vector<NSBRKGA::Sense> senses = {NSBRKGA::Sense::MINIMIZE,
                                           NSBRKGA::Sense::MINIMIZE};
    auto fronts = NSBRKGA::Population::nonDominatedSort<unsigned>(fitness, senses);
    check("single element → 1 front", fronts.size() == 1);
    check("single element → front has 1 entry", fronts[0].size() == 1);
}

static void test_1obj_minimize() {
    std::cout << "[test_1obj_minimize]" << std::endl;
    // 1.0, 2.0, 2.0, 3.0 → front0={1.0}, front1={2.0, 2.0}, front2={3.0}
    std::vector<Pair> fitness = {
        {{2.0}, 1}, {{3.0}, 3}, {{1.0}, 0}, {{2.0}, 2}};
    std::vector<NSBRKGA::Sense> senses = {NSBRKGA::Sense::MINIMIZE};
    auto fronts = NSBRKGA::Population::nonDominatedSort<unsigned>(fitness, senses);

    check("1obj min: 3 fronts", fronts.size() == 3);
    check("1obj min: front0 size=1", fronts[0].size() == 1);
    check("1obj min: front0 val=1.0", fronts[0][0].first[0] == 1.0);
    check("1obj min: front1 size=2", fronts[1].size() == 2);
    check("1obj min: front1 vals=2.0",
          fronts[1][0].first[0] == 2.0 && fronts[1][1].first[0] == 2.0);
    check("1obj min: front2 size=1", fronts[2].size() == 1);
    check("1obj min: front2 val=3.0", fronts[2][0].first[0] == 3.0);
}

static void test_1obj_maximize() {
    std::cout << "[test_1obj_maximize]" << std::endl;
    // After sort (max): 5.0, 3.0, 3.0, 1.0
    // front0={5.0}, front1={3.0, 3.0}, front2={1.0}
    std::vector<Pair> fitness = {
        {{3.0}, 1}, {{1.0}, 3}, {{5.0}, 0}, {{3.0}, 2}};
    std::vector<NSBRKGA::Sense> senses = {NSBRKGA::Sense::MAXIMIZE};
    auto fronts = NSBRKGA::Population::nonDominatedSort<unsigned>(fitness, senses);

    check("1obj max: 3 fronts", fronts.size() == 3);
    check("1obj max: front0 val=5.0", fronts[0][0].first[0] == 5.0);
    check("1obj max: front1 size=2", fronts[1].size() == 2);
    check("1obj max: front2 val=1.0", fronts[2][0].first[0] == 1.0);
}

static void test_2obj_minimize() {
    std::cout << "[test_2obj_minimize]" << std::endl;
    //  A=(1,4)  B=(2,3)  C=(3,2)  D=(4,1)  E=(2,4)  F=(3,3)
    //  Front 0: {A, B, C, D}  (non-dominated)
    //  Front 1: {E, F}
    std::vector<Pair> fitness = {
        {{1.0, 4.0}, 0}, {{2.0, 3.0}, 1}, {{3.0, 2.0}, 2},
        {{4.0, 1.0}, 3}, {{2.0, 4.0}, 4}, {{3.0, 3.0}, 5}};
    std::vector<NSBRKGA::Sense> senses = {NSBRKGA::Sense::MINIMIZE,
                                           NSBRKGA::Sense::MINIMIZE};
    auto fronts = NSBRKGA::Population::nonDominatedSort<unsigned>(fitness, senses);

    check("2obj min: 2 fronts", fronts.size() == 2);
    check("2obj min: front0 size=4", fronts[0].size() == 4);
    check("2obj min: front1 size=2", fronts[1].size() == 2);
}

static void test_2obj_mixed_senses() {
    std::cout << "[test_2obj_mixed_senses]" << std::endl;
    // obj1 MINIMIZE, obj2 MAXIMIZE
    // A=(1,3) B=(2,4) C=(1,4) → C dominates A; B and C non-dominated
    std::vector<Pair> fitness = {
        {{1.0, 3.0}, 0}, {{2.0, 4.0}, 1}, {{1.0, 4.0}, 2}};
    std::vector<NSBRKGA::Sense> senses = {NSBRKGA::Sense::MINIMIZE,
                                           NSBRKGA::Sense::MAXIMIZE};
    auto fronts = NSBRKGA::Population::nonDominatedSort<unsigned>(fitness, senses);

    check("2obj mixed: 2 fronts", fronts.size() == 2);
    // C=(1,4) dominates A=(1,3) and B=(2,4), so front0={C}, front1={A,B}
    check("2obj mixed: front0 size=1", fronts[0].size() == 1);
    check("2obj mixed: front1 size=2", fronts[1].size() == 2);
}

static void test_3obj() {
    std::cout << "[test_3obj]" << std::endl;
    // A=(1,2,3) B=(2,1,3) C=(3,3,1) → all non-dominated by each other
    // D=(2,3,4) → dominated by A
    std::vector<Pair> fitness = {
        {{1.0, 2.0, 3.0}, 0}, {{2.0, 1.0, 3.0}, 1},
        {{3.0, 3.0, 1.0}, 2}, {{2.0, 3.0, 4.0}, 3}};
    std::vector<NSBRKGA::Sense> senses = {NSBRKGA::Sense::MINIMIZE,
                                           NSBRKGA::Sense::MINIMIZE,
                                           NSBRKGA::Sense::MINIMIZE};
    auto fronts = NSBRKGA::Population::nonDominatedSort<unsigned>(fitness, senses);

    check("3obj: ≥2 fronts", fronts.size() >= 2);
    // D should not be in front 0
    bool d_in_front0 = false;

    for (auto &p : fronts[0]) {
        if (p.second == 3) {
            d_in_front0 = true;
        }
    }

    check("3obj: D not in front 0", !d_in_front0);

    // All of A, B, C should be in front 0
    unsigned count_abc = 0;

    for (auto &p : fronts[0]) {
        if (p.second <= 2) {
            count_abc++;
        }
    }

    check("3obj: A, B, C in front 0", count_abc == 3);
}

static void test_ties_identical() {
    std::cout << "[test_ties_identical]" << std::endl;
    // All identical → all in one front
    std::vector<Pair> fitness = {
        {{2.0, 3.0}, 0}, {{2.0, 3.0}, 1}, {{2.0, 3.0}, 2}};
    std::vector<NSBRKGA::Sense> senses = {NSBRKGA::Sense::MINIMIZE,
                                           NSBRKGA::Sense::MINIMIZE};
    auto fronts = NSBRKGA::Population::nonDominatedSort<unsigned>(fitness, senses);

    check("ties: 1 front", fronts.size() == 1);
    check("ties: all 3 in front 0", fronts[0].size() == 3);
}

static void test_all_dominated_chain() {
    std::cout << "[test_all_dominated_chain]" << std::endl;
    // Total domination chain: (1,1) > (2,2) > (3,3)
    std::vector<Pair> fitness = {
        {{3.0, 3.0}, 2}, {{1.0, 1.0}, 0}, {{2.0, 2.0}, 1}};
    std::vector<NSBRKGA::Sense> senses = {NSBRKGA::Sense::MINIMIZE,
                                           NSBRKGA::Sense::MINIMIZE};
    auto fronts = NSBRKGA::Population::nonDominatedSort<unsigned>(fitness, senses);

    check("chain: 3 fronts", fronts.size() == 3);
    check("chain: front0 size=1", fronts[0].size() == 1);
    check("chain: front1 size=1", fronts[1].size() == 1);
    check("chain: front2 size=1", fronts[2].size() == 1);
}

// ---------------------------------------------------------------------------

int main() {
    std::cout << "=== nonDominatedSort regression tests ===" << std::endl;
    test_empty_fitness();
    test_empty_senses();
    test_single_element();
    test_1obj_minimize();
    test_1obj_maximize();
    test_2obj_minimize();
    test_2obj_mixed_senses();
    test_3obj();
    test_ties_identical();
    test_all_dominated_chain();

    std::cout << "\n--- Summary: " << pass_count << "/" << test_count
              << " passed, " << fail_count << " failed ---" << std::endl;

    return fail_count > 0 ? 1 : 0;
}
