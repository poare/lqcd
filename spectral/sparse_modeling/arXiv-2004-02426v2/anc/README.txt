
Sparse modeling approach to analytic continuation
    of imaginary-time Monte Carlo data
                    3 Apr 2020  E.Itou and Y. Nagai

LICENSE: MIT LICENSE (see LICENSE.txt)

The medhod is written in four languages, FORTRAN, Fortran, Julia and c++. 
The version: 
Fortran: Fortran2003 or higher (We checked it with gfortran in gcc version 9.1.0)
Julia: Julia 1.3 or higher
c++: c++17 is used (We checked it with g++ in gcc version 9.1.0).
FORTRAN; FORTRAN77 is used (limited. We commend to use the Fortran2003 version.)

Julia version is recommended since this can automatticaly generates the kernel even if there is no file about the kernel. 

How to use:
In SpM directory: There are sparse modeling codes written in four languages.

Fortran:
make
./sparse input.in
c++:
sh compile.sh
./sparse_c input.in
Julia:
julia main_v2.jl input.in
In julia version, we can make the kernel in main_v2.jl. 
In other versions, we can not make the kernel. So we have to make the kernel BEFORE doing a calculation.

In MakeKernel directory: There is a code to make the kernel written in Julia.
julia kinput.in

-------------------------
in input.in and kinput.in, 
NT_org is the total number of the imaginary-time points.
NT is the number of the truncated imaginary-time pointes. This should be odd. 

NT_org and NT should be same in both input.in and kinput.in

Nomega is the number of the frequency points. 
Ctau is the filename of the correlation function.


