#ifndef RBF_RBF_H
#define RBF_RBF_H

#include <iostream>
#include <armadillo>
#include <vector>
#include <fstream>
#include <algorithm>
#include <cstdlib>
#include <math.h>

class RBF
{
public:
    RBF(int n_cent, double sig);

    void fit(char * xobs_path, char * yobs_path, int row, int col);
    void model(char * xmod_path, char * ymod_path, int row, int col);

    void set_interpol_matrix(arma::Mat <double> &interpol_mtx, arma::Mat <double> X, int row);

    void load_data(std::vector < std::vector <double> > &INP, char * inp_path, int row, int col);
    void load_data(std::vector <double> &INP, char * inp_path, int row);

    void stand(std::vector < std::vector <double> > & input, std::vector <double> & mean, std::vector <double> & stdev);
    void apply_stand(std::vector < std::vector <double> > & input, std::vector <double> & mean, std::vector <double> & stdev);


private:
    int n_centers;
    double sigma;

    arma::Mat<double> centers;
    arma::Mat<double> weights;

    std::vector <double> mean;
    std::vector <double> stdev;
};


arma::mat vectomat(std::vector < std::vector <double> > &inp);





#endif //RBF_RBF_H
