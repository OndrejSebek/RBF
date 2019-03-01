#include "RBF.h"

RBF::RBF(int n_cent, double sig)
{
    std::srand(time(NULL));
    n_centers = n_cent;
    sigma = sig;
}

void RBF::fit(char * xobs_path, char * yobs_path, int row, int col)
{
    // load data
    std::vector < std::vector <double> > X_obs;
    std::vector <double> Y_obs;

    mean.resize(col);
    stdev.resize(col);

    load_data(X_obs, xobs_path, row, col);
    load_data(Y_obs, yobs_path, row);

    stand(X_obs, mean, stdev);

    arma::Mat<double> X_obs_mat = vectomat(X_obs);
    arma::Mat<double> Y_obs_mat = arma::conv_to<arma::mat>::from(Y_obs);

    // select centers (random)

        // random indices
    std::vector < int > center_ind(row);
    for(int i = 0; i < row; i++)
        center_ind[i] = i;

    std::random_shuffle(center_ind.begin(), center_ind.end());

        // centers
    centers = arma::mat(n_centers, col);

    for(int i = 0; i < n_centers; i++)
        centers.row(i) = X_obs_mat.row(center_ind[i]);


    // calc interpolation mtx
    arma::Mat<double> interpol_matrix(row, n_centers);
    set_interpol_matrix(interpol_matrix, X_obs_mat, row);


    // calc weights
    weights = arma::pinv(interpol_matrix) * Y_obs_mat;
}

void RBF::set_interpol_matrix(arma::Mat <double> &interpol_mtx, arma::Mat <double> X, int row)
{
    double dist;

    for(int i = 0; i < row; i++)
    {
        for(int j = 0; j < n_centers; j++)
        {
            // kernel
            dist = arma::norm(centers.row(j) - X.row(i));
            interpol_mtx.row(i)[j] = exp(-sigma*dist*dist);
        }
    }
}

void RBF::model(char * xmod_path, char * ymod_path, int row, int col)
{
    // load data
    std::vector < std::vector <double> > X_mod;

    load_data(X_mod, xmod_path, row, col);

    apply_stand(X_mod, mean, stdev);

    arma::Mat<double> X_mod_mat = vectomat(X_mod);


    // calc interpolation matrix
    arma::Mat<double> interpol_matrix(row, n_centers);
    set_interpol_matrix(interpol_matrix, X_mod_mat, row);


    // model Y_mod
    arma::Mat<double> Y_mod_mtx;

    Y_mod_mtx = interpol_matrix * weights;
    Y_mod_mtx.save(ymod_path, arma::csv_ascii);
}



arma::mat vectomat(std::vector < std::vector <double> > &inp)
{
    arma::mat A = arma::mat(inp.size(), inp[0].size());
    for(int i = 0; i < inp.size(); i++)
    {
        A.row(i) = arma::conv_to<arma::rowvec>::from(inp[i]);
    }

    return A;
}


void RBF::load_data(std::vector < std::vector <double> > &INP, char * inp_path, int row, int col)
{
    INP.resize(row);
    for(int i = 0; i < row; i++)
        INP[i].resize(col);

    std::ifstream inp_file;
    inp_file.open(inp_path);

    for(int i = 0; i < row; i++)
        for(int j = 0; j < col; j++)
            inp_file >> INP[i][j];
}

void RBF::load_data(std::vector <double> &INP, char * inp_path, int row)
{
    INP.resize(row);

    std::ifstream inp_file;
    inp_file.open(inp_path);

    for(int i = 0; i < row; i++)
        inp_file >> INP[i];
}



void RBF::stand(std::vector < std::vector <double> > & input, std::vector <double> & mean, std::vector <double> & stdev)
{
    // MEAN
    for(int i = 0; i < input.size(); i++)
    {
        for(int j = 0; j < input[0].size(); j++)
        {
            mean[j] += input[i][j];
        }
    }
    for(int i = 0; i < input[0].size(); i++)
        mean[i] /= input.size();

    // STDEV
    for(int i = 0; i < input.size(); i++)
    {
        for(int j = 0; j < input[0].size(); j++)
        {
            stdev[j] += (input[i][j] - mean[j])*(input[i][j] - mean[j]);
        }
    }
    for(int i = 0; i < input[0].size(); i++)
        stdev[i] = sqrt(stdev[i] / input.size());

    // -> STAND
    for(int i = 0; i < input.size(); i++)
    {
        for(int j = 0; j < input[0].size(); j++)
        {
            input[i][j] = input[i][j] / stdev[j] - mean[j];
        }
    }
}

void RBF::apply_stand(std::vector < std::vector <double> > & input, std::vector <double> & mean, std::vector <double> & stdev)
{
    for(int i = 0; i < input.size(); i++)
    {
        for(int j = 0; j < input[0].size(); j++)
        {
            input[i][j] = input[i][j] / stdev[j] - mean[j];
        }
    }
}