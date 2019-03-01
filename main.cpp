#include <iostream>
#include <armadillo>
#include <fstream>
#include <vector>

#include "RBF.h"


int main()
{
    RBF rbf(50, .8);

    rbf.fit("data/inp3/X_obs.txt", "data/inp3/Y_obs.txt", 2265, 3);

    rbf.model("data/inp3/X_mod.txt", "out/Y_mod_3.txt", 1510, 3);


    return 0;
}