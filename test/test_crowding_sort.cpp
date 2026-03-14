/**
 * Regression test for Population::crowdingSort.
 *
 * Exercises single-element, two-element, degenerate-range,
 * typical 2-obj and 3-obj fronts, and tie-breaking.
 *
 * Build:
 *   g++ -std=c++14 -I../nsbrkga test_crowding_sort.cpp -o test-crowding-sort
 * Run:
 *   ./test-crowding-sort
 */

#include "nsbrkga.hpp"

#include <cassert>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <string>
#include <utility>
#include <vector>

using Pair = std::pair<std::vector<double>, unsigned>;
using Front = std::vector<Pair>;

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

// Helper: extract payload (original index) vector.
static std::vector<unsigned> payloadsOf(const Front &f) {
    std::vector<unsigned> r;
    r.reserve(f.size());

    for (auto &p : f) {
        r.push_back(p.second);
    }

    return r;
}

// ---- Test cases -----------------------------------------------------------

static void test_single_element() {
    std::cout << "[test_single_element]" << std::endl;
    Front front = {{{1.0, 2.0}, 0}};
    NSBRKGA::Population::crowdingSort<unsigned>(front);
    check("single: size unchanged", front.size() == 1);
    check("single: payload preserved", front[0].second == 0);
}

static void test_two_elements() {
    std::cout << "[test_two_elements]" << std::endl;
    // Both boundary → both get max(), tie-broken by descending index.
    Front front = {{{1.0, 4.0}, 0}, {{2.0, 3.0}, 1}};
    NSBRKGA::Population::crowdingSort<unsigned>(front);
    check("two: size unchanged", front.size() == 2);
    // Both should have max distance; order depends on tie-breaking.
    // We just verify both are present.
    auto ids = payloadsOf(front);
    bool has0 = (ids[0] == 0 || ids[1] == 0);
    bool has1 = (ids[0] == 1 || ids[1] == 1);
    check("two: both payloads present", has0 && has1);
}

static void test_all_identical_objectives() {
    std::cout << "[test_all_identical_objectives]" << std::endl;
    // Degenerate range → all get max() distance.
    Front front = {
        {{3.0, 3.0}, 0}, {{3.0, 3.0}, 1}, {{3.0, 3.0}, 2}};
    NSBRKGA::Population::crowdingSort<unsigned>(front);
    check("identical: size unchanged", front.size() == 3);
    // All three should be present (order is tie-breaking dependent).
    auto ids = payloadsOf(front);
    bool ok = true;

    for (unsigned x = 0; x < 3; x++) {
        bool found = false;

        for (auto id : ids) if (id == x) {
            found = true;
        }

        if (!found) {
            ok = false;
        }
    }

    check("identical: all payloads present", ok);
}

static void test_2obj_typical_front() {
    std::cout << "[test_2obj_typical_front]" << std::endl;
    // A non-dominated front for minimisation: (1,4) (2,3) (3,2) (4,1)
    // After crowding sort, boundary solutions (indices 0 and 3) should
    // appear first because they have max() distance.
    Front front = {
        {{1.0, 4.0}, 0}, {{2.0, 3.0}, 1},
        {{3.0, 2.0}, 2}, {{4.0, 1.0}, 3}};

    // Keep a copy to verify against.
    Front original = front;
    NSBRKGA::Population::crowdingSort<unsigned>(front);

    check("2obj: size unchanged", front.size() == 4);

    // The first two positions should be boundary solutions (payloads 0 & 3)
    // because they have infinite distance.
    auto ids = payloadsOf(front);
    bool first_is_boundary = (ids[0] == 0 || ids[0] == 3);
    bool second_is_boundary = (ids[1] == 0 || ids[1] == 3);
    check("2obj: boundaries first", first_is_boundary && second_is_boundary);

    // Interior solutions (payloads 1 & 2) have equal finite distance
    // (each spans the same normalised range), so they come last.
    bool third_is_interior = (ids[2] == 1 || ids[2] == 2);
    bool fourth_is_interior = (ids[3] == 1 || ids[3] == 2);
    check("2obj: interiors last", third_is_interior && fourth_is_interior);

    // All payloads preserved.
    bool all_present = true;

    for (unsigned x = 0; x < 4; x++) {
        bool found = false;
        for (auto id : ids) {
            if (id == x) {
                found = true;
            }
        }

        if (!found) {
            all_present = false;
        }
    }

    check("2obj: all payloads present", all_present);
}

static void test_3obj_front() {
    std::cout << "[test_3obj_front]" << std::endl;
    // 5 solutions with 3 objectives.
    Front front = {
        {{1.0, 5.0, 3.0}, 0},
        {{2.0, 4.0, 2.0}, 1},
        {{3.0, 3.0, 1.0}, 2},
        {{4.0, 2.0, 4.0}, 3},
        {{5.0, 1.0, 5.0}, 4}};
    NSBRKGA::Population::crowdingSort<unsigned>(front);
    check("3obj: size unchanged", front.size() == 5);

    // All payloads still present.
    auto ids = payloadsOf(front);
    bool all_present = true;

    for (unsigned x = 0; x < 5; x++) {
        bool found = false;
        for (auto id : ids) {
            if (id == x) {
                found = true;
            }
        }

        if (!found) {
            all_present = false;
        }
    }

    check("3obj: all payloads present", all_present);
}

static void test_empty_front() {
    std::cout << "[test_empty_front]" << std::endl;
    Front front;
    // Should not crash.
    NSBRKGA::Population::crowdingSort<unsigned>(front);
    check("empty: size is 0", front.empty());
}

static void test_consistency_with_sort_fitness() {
    std::cout << "[test_consistency_with_sort_fitness]" << std::endl;
    // Build a small 2-objective population and verify that sortFitness
    // (which calls nonDominatedSort then crowdingSort) produces a valid
    // result.
    using Fitness = std::vector<std::pair<std::vector<double>, unsigned>>;
    Fitness fitness = {
        {{1.0, 6.0}, 0}, {{2.0, 5.0}, 1}, {{3.0, 4.0}, 2},
        {{4.0, 3.0}, 3}, {{5.0, 2.0}, 4}, {{6.0, 1.0}, 5},
        {{2.0, 6.0}, 6}, {{3.0, 5.0}, 7}};
    std::vector<NSBRKGA::Sense> senses = {NSBRKGA::Sense::MINIMIZE,
                                           NSBRKGA::Sense::MINIMIZE};
    auto result =
        NSBRKGA::Population::sortFitness<unsigned>(fitness, senses);
    check("sortFitness: num_fronts > 0", result.first > 0);
    check("sortFitness: num_non_dominated > 0", result.second > 0);
    check("sortFitness: size preserved", fitness.size() == 8);

    // Verify all original payloads are present.
    bool all_present = true;
    for (unsigned x = 0; x < 8; x++) {
        bool found = false;

        for (auto &p : fitness) {
            if (p.second == x) {
                found = true;
            }
        }

        if (!found) {
            all_present = false;
        }
    }

    check("sortFitness: all payloads present", all_present);
}

// ---------------------------------------------------------------------------

int main() {
    std::cout << "=== crowdingSort regression tests ===" << std::endl;
    test_empty_front();
    test_single_element();
    test_two_elements();
    test_all_identical_objectives();
    test_2obj_typical_front();
    test_3obj_front();
    test_consistency_with_sort_fitness();

    std::cout << "\n--- Summary: " << pass_count << "/" << test_count
              << " passed, " << fail_count << " failed ---" << std::endl;

    return fail_count > 0 ? 1 : 0;
}
