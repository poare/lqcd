
//***********************************************************
// Sparse modeling approach to analytic continuation
//      of imaginary-time Monte Carlo data
//                           3 Apr 2020  E. Itou and Y. Nagai
//***********************************************************



#include "input.cpp"
#include "smac_lib.cpp"

#include <iostream>
#include <string>
#include <sstream>
#include <iomanip>
#include <tuple>
#include <Eigen/Core>
#include <vector>
#define PRINT_MAT(X) std::cout << #X << ":\n" << X << std::endl << std::endl


double 
    loadK(const int NT, const int Nomega, const std::string filename,Eigen::MatrixXd& mat_K ) {
        std::string tmp;

        double Omega_max = 0.0;
        mat_K.fill(0.0);

        std::ifstream ifs(filename, std::ios::in);
        getline(ifs, tmp);
        std::stringstream ss;
        ss << tmp;
        ss >> Omega_max;
        std::cout << Omega_max << std::endl;
        for (int it = 0; it < NT; it++){
            for (int io = 0; io < Nomega; io++){
                getline(ifs, tmp);
                std::stringstream ss;
                ss << tmp;
                ss >> mat_K(it,io);
                //std::cout << mat_K(it,io)  << std::endl;
            };
        };
    return Omega_max;
}

Eigen::VectorXd 
    loadG(const int NT, const std::string filename) {
        Eigen::VectorXd vec_G(NT);
        std::string tmp;
        vec_G.fill(0.0);
        double dum1;
        double dum2;
        double dum3;

        std::ifstream ifs(filename, std::ios::in);

        for (int it = 0; it < NT; it++){
            getline(ifs, tmp);
            std::stringstream ss;
            ss << tmp;
            ss >> dum1 >> dum2 >> vec_G(it) >> dum3;
            //std::cout << vec_G(it)  << std::endl;
        };
    return vec_G;
}

Eigen::VectorXd 
    load1d(const int NT, const std::string filename) {
        Eigen::VectorXd vec_val(NT);
        std::string tmp;
        vec_val.fill(0.0);

        std::ifstream ifs(filename, std::ios::in);

        for (int it = 0; it < NT; it++){
            getline(ifs, tmp);
            std::stringstream ss;
            ss << tmp;
            ss >>  vec_val(it) ;
            //std::cout << vec_val(it)  << std::endl;
        };
    return vec_val;
}

Eigen::MatrixXd 
    load2d(const int NT, const int Nomega, const std::string filename) {
        Eigen::MatrixXd mat_val(NT,Nomega);
        std::string tmp;

        mat_val.fill(0.0);

        std::ifstream ifs(filename, std::ios::in);
        for (int it = 0; it < NT; it++){
            for (int io = 0; io < Nomega; io++){
                getline(ifs, tmp);
                std::stringstream ss;
                ss << tmp;
                ss >> mat_val(it,io);
                //std::cout << mat_K(it,io)  << std::endl;
            };
        };
    return mat_val;
}



int main(int argc, char* argv[]){
    Input* input;
    input = new Input(argc,argv);

    auto NT = input->getInt("NT",10);
    
    auto Nomega = input->getInt("Nomega",2001);
    auto NT_org = input->getInt("NT_org",10);

    std::cout << "NT = " << NT << std::endl; 
    std::cout << "Nomega = " << Nomega << std::endl; 
    std::cout << "NT_org = " << NT_org << std::endl; 

    std::ostringstream sout;
    sout << std::setfill('0') << std::setw(6) << Nomega;
    std::string inomega = sout.str();
    std::cout << inomega << std::endl;

    double domega = 2.0/(Nomega-1);
    std::string RFILE = "./data-K-Nt"+std::to_string(NT_org)+"-Ndomega"+inomega+".dat";
    std::cout << "read file " << RFILE << std::endl;


    Eigen::MatrixXd mat_K(NT,Nomega);
    double Omega_max = 0.0;
    if(checkFileExistence(RFILE)){
        std::cout << "The kernel K is loaded" << std::endl;
        Omega_max = loadK(NT, Nomega,RFILE,mat_K);
        
    }else{
        std::cout << "Making the kernel K..." << std::endl;
    }
    //PRINT_MAT(mat_K);

    RFILE =input->getstring("Ctau","test.dat");
    std::cout <<  RFILE << std::endl;
    auto vec_G = loadG(NT,RFILE);
    auto d_norm = vec_G(0)/sqrt(domega);
    std::cout << "d_norm " << d_norm << std::endl;
    for (int il = 0; il < NT; il++){
        vec_G(il) = vec_G(il)/d_norm;
    }
    

    RFILE = "./data-s-value"+inomega+".dat";
    auto data_S = load1d(NT,RFILE);

    RFILE = "./data-matrix-U"+inomega+".dat";
    auto U = load2d(NT,NT,RFILE);

    RFILE = "./data-matrix-V"+inomega+".dat";
    auto V = load2d(NT,Nomega,RFILE);
    Eigen::MatrixXd V_t = V.transpose();
    std::cout <<  "read file done"<< std::endl;

    int ii = 0;
    for (int it = 0; it < NT; it++){
        ii = ii + 1;
        if(data_S(it) < 1e-10){
            break;
        }
    };
    int L = ii;
    std::cout <<  "L = " << L << std::endl;
    //PRINT_MAT(data_S);
    //PRINT_MAT(U);

    auto tempvec= data_S.block(0,0,L,1);
    data_S = tempvec;
    //PRINT_MAT(data_S);


    auto tempmat = U.block(0,0,NT,L);
    U = tempmat;
    //PRINT_MAT(U);

    auto tempmat2 = V_t.block(0,0,Nomega,L);
    V_t = tempmat2;
    //PRINT_MAT(V_t);
    V = V_t.transpose();
    //PRINT_MAT(V);
    std::cout <<  "mat_reduced done"<< std::endl;

    Eigen::VectorXd e(Nomega);
    e.fill(1.0);
    
    auto xi = V*e;
    //PRINT_MAT(xi);
    for (int il = 0; il < L; il++){
        std::cout <<  "xi in main "<< il << " " << xi(il) << std::endl;
    }

    std::vector<double> vec_Gout(NT,0.0);
    std::vector<double> xout(Nomega,0.0);

    Smac* smac;
    smac = new Smac(NT,Nomega, L, mat_K,U,V,V_t,data_S);
    //PRINT_MAT(smac->vec_S);
    
    std::vector<double> vec_Gv(NT);
    for (int il = 0; il < NT; il++){
        vec_Gv[il] = vec_G(il);
    }

    std::vector<double> xiv(L);
    for (int il = 0; il < L; il++){
        xiv[il] = xi(il);
    }

    std::cout <<  "before smac " << std::endl;
    //PRINT_MAT(smac->mat_K);
    xout = smac->smac_est(vec_Gv, xiv);
    std::cout <<  "after smac " << std::endl;

    std::string WFILE = "./data-rho-Nomega"+inomega+".dat";

    std::ofstream writeFile;
    writeFile.open(WFILE);
    for (int io = 0; io < Nomega; io++){
        writeFile << (io*domega-1)*Omega_max/NT_org << " " << xout[io]*d_norm/sqrt(domega)/std::pow(NT_org,4)/Omega_max << std::endl;
    }
    writeFile.close();

    std::ofstream writeFile2;
    WFILE = "./data-C_tau_result-Nomega"+inomega+".dat";
    //std::cout <<  WFILE << std::endl;
    writeFile2.open(WFILE);
    for (int it = 0; it < NT; it++){
        for (int io = 0; io < Nomega; io++){
            vec_Gout[it] = vec_Gout[it]+mat_K(it,io)*xout[io]*d_norm;
        }
        writeFile2 << "it,vec_G " << " " << it << " " << vec_Gout[it] << std::endl;
    }
    writeFile2.close();

    return 0;
};