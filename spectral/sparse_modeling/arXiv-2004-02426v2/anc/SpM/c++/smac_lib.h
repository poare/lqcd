// #define EIGEN_RUNTIME_NO_MALLOC 
#include <Eigen/Core>
#include <vector> 

class Smac{
    public:
    int NT;
    int Nomega;
    int L;
    Eigen::MatrixXd mat_K;
    Eigen::MatrixXd U;
    Eigen::MatrixXd V;
    Eigen::MatrixXd V_t;
    Eigen::VectorXd vec_S;

    Smac(const int NT,const int Nomega,const int L, const Eigen::MatrixXd& mat_K,const Eigen::MatrixXd& U,
        const Eigen::MatrixXd& V, const Eigen::MatrixXd& V_t,const Eigen::VectorXd& vec_S
    );


    std::vector<double> smac_est(const std::vector<double>& vec_G, const std::vector<double>& xi);


    std::vector<double> smac_calc(const std::vector<double>& y,const std::vector<double>& xi,const int itemax,
const double lambda,const double d_mu,const double d_mup);

    void smac_update(const std::vector<double> & styp, 
    std::vector<double> & xp,std::vector<double> & zp,std::vector<double> & up,
    std::vector<double> & z,std::vector<double> & u,const std::vector<double> & xi,
    const double lambda,const double d_mu,const double d_mup);
};