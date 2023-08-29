//***********************************************************
// Sparse modeling approach to analytic continuation
//      of imaginary-time Monte Carlo data
//                           3 Apr 2020  E. Itou and Y. Nagai
//***********************************************************

#include <iostream>
#include <fstream>
#include <sstream>

#include "input.h"


Input::Input(int argc, char* argv[]){
    using namespace std;
    if(argc != 2){
        cerr << "num of parameters is wrong." << endl;
        cerr << "./sparse_c input.in" << endl;
    }

    _filename = argv[1];
    _ifs.open(_filename, std::ios::in);
    std::cout << _filename << std::endl; 
    if(!_ifs){
        std::cerr << "Error: file not opened." << std::endl;
    }

};

int Input::getInt(std::string name,int initialvalue){
    std::string tmp;
    std::string strname;
    std::string strtmp;
    int num;
    

    while(getline(_ifs, tmp)){
        //std::cout << tmp << std::endl; 
        std::stringstream ss;
        ss << tmp;
        ss >> strname >> strtmp >> num; 
        if(strname == name){
            _ifs.close();
            _ifs.open(_filename, std::ios::in);
            return num;
//            std::cout << strname << "nameok" << std::endl; 
        }
        
        //std::cout << num << std::endl; 
    }
    _ifs.close();
    _ifs.open(_filename, std::ios::in);
    //_ifs.close();

    return initialvalue;
}

std::string Input::getstring(std::string name,std::string initialvalue){
    std::string tmp;
    std::string strname;
    std::string strtmp;
    std::string st;

    while(getline(_ifs, tmp)){
        std::cout << tmp << std::endl; 
        std::stringstream ss;
        ss << tmp;
        ss >> strname >> strtmp >> st; 
        if(strname == name){
            _ifs.close();
            _ifs.open(_filename, std::ios::in);
            return st;
//            std::cout << strname << "nameok" << std::endl; 
        }
        
        //std::cout << st << std::endl; 
    }
    _ifs.close();
    _ifs.open(_filename, std::ios::in);
    

    return initialvalue;
}

bool checkFileExistence(const std::string& str)
{
    std::ifstream ifs(str);
    return ifs.is_open();
}

