#include "decoders.hpp"

#include <iostream>
using namespace std;

std::vector<double> Sum_Decoder::decode(BRKGA::Chromosome& chromosome, bool /*non-used*/) {
    // Just sum the values
    double total = 0.0;
    for(auto &v : chromosome) {
        total += v;
    }

    return std::vector<double>(1, total);
}

std::vector<double> Order_Decoder::decode(BRKGA::Chromosome& chromosome, bool /*non-use*/) {
    // Just sum the values
    double total = 0.0;
    double last = chromosome.front();
    for(const auto &v : chromosome) {
        if(last < v) {
            total += 1.0;
        }
        last = v;
    }

    return std::vector<double>(1, total);
}
