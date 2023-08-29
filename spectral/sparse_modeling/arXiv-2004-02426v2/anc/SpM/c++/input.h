#include <string>
#include <iostream>
#include <fstream>
#include <sstream>

class Input{
    private:
    std::string _filename;
    std::ifstream _ifs;
    

    public:
    
    Input(int argc, char* argv[]);
    int getInt(std::string name,int initialvalue);
    std::string getstring(std::string name,std::string initialvalue);

};