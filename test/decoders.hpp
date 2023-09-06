#include "chromosome.hpp"

//---------------------------[ Dummy decoder ]--------------------------------//

class Sum_Decoder {
    public:
        std::vector<double> decode(NSBRKGA::Chromosome& chromosome, bool foo = true);
};

class Order_Decoder {
    public:
        std::vector<double> decode(NSBRKGA::Chromosome& chromosome, bool foo = true);
};
