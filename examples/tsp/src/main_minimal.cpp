/******************************************************************************
 * main_minimal.cpp: minimal code for calling BRKGA algorithms to solve
 *                   instances of the Traveling Salesman Problem.
 *
 * (c) Copyright 2015-2019, Carlos Eduardo de Andrade.
 * All Rights Reserved.
 *
 *  Created on : Mar 05, 2019 by andrade
 *  Last update: Jul 02, 2020 by luishpmendes
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
 *****************************************************************************/

#include "tsp/tsp_instance.hpp"
#include "decoders/tsp_decoder.hpp"
#include "nsbrkga.hpp"

#include <iostream>
#include <stdexcept>
#include <string>

using namespace std;

//-------------------------------[ Main ]------------------------------------//

int main(int argc, char* argv[]) {
    if(argc < 4) {
        cerr << "Usage: "<< argv[0]
             << " <seed> <config-file> <num-generations>"
                " <tsp-instance-file>" << endl;
        return 1;
    }

    try {
        ////////////////////////////////////////
        // Read command-line arguments and the instance
        ////////////////////////////////////////

        const unsigned seed = stoi(argv[1]);
        const string config_file = argv[2];
        const unsigned num_generations = stoi(argv[3]);
        const string instance_file = argv[4];

        cout << "Reading data..." << endl;
        auto instance = TSP_Instance(instance_file);

        ////////////////////////////////////////
        // Read algorithm parameters
        ////////////////////////////////////////

        cout << "Reading parameters..." << endl;

        // C++14 syntax.
        auto params = BRKGA::readConfiguration(config_file);
        auto& brkga_params = params.first;

        // C++17 syntax.
        // auto [brkga_params, control_params] =
        //     BRKGA::readConfiguration(config_file);

        ////////////////////////////////////////
        // Build the BRKGA data structures and initialize
        ////////////////////////////////////////

        cout << "Building BRKGA data and initializing..." << endl;

        TSP_Decoder decoder(instance);

        BRKGA::NSBRKGA<TSP_Decoder> algorithm(decoder,
                std::vector<BRKGA::Sense>(1, BRKGA::Sense::MINIMIZE), seed,
                instance.num_nodes, brkga_params);

        // NOTE: don't forget to initialize the algorithm.
        algorithm.initialize();

        ////////////////////////////////////////
        // Find good solutions / evolve
        ////////////////////////////////////////

        cout << "Evolving " << num_generations << " generations..." << endl;
        algorithm.evolve(num_generations);

        auto best_cost = algorithm.getIncumbentFitnesses()[0][0];
        cout << "Best cost: " << best_cost << endl;
    }
    catch(exception& e) {
        cerr << "\n***********************************************************"
             << "\n****  Exception Occured: " << e.what()
             << "\n***********************************************************"
             << endl;
        return 70; // BSD software internal error code
    }
    return 0;
}
