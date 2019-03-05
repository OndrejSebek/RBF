#include <iostream>
#include <armadillo>
#include <fstream>
#include <vector>

#include "RBF.h"


int main()
{
    RBF rbf(50, .2);                                                            // n_centers, sigma

    rbf.fit("data/inp3/X_obs.txt", "data/inp3/Y_obs.txt", 2265, 3);             // obs_inp fpath, obs_exp_out fpath, row, col

    rbf.model("data/inp3/X_mod.txt", "out/Y_mod_3.txt", 1510, 3);               // mod_inp fpath, mod_out fpath, row, col


    return 0;
}
