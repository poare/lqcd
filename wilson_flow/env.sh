##### 
# SET UP ENVIRONMENT
module load gcc/5.4.0
module load openmpi/2.1.1
export GCC_HOME=/opt/software/gcc-5.4.0/

OMP="yes"
SM=sm_61
export CUDA_INSTALL_PATH=/opt/software/cuda-8.0
export PATH=/data/wombat/users/mlwagman/utils/cmake-3.10.2/bin:$PATH
export PATH=${CUDA_INSTALL_PATH}/bin:${MPI_HOME}/bin:$PATH
export LD_LIBRARY_PATH=${CUDA_INSTALL_PATH}/lib64:${CUDA_INSTALL_PATH}/lib:${CUDA_INSTALL_PATH}/nvvm/lib64:/data/wombat/users/mlwagman/utils/lapack-3.8.0/build/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=${MPI_HOME}/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=${GCC_HOME}/lib:$LD_LIBRARY_PATH
export PATH=${GCC_HOME}:$PATH

# The directory containing the build scripts, this script and the src/ tree
TOPDIR=/data/wombat/users/mlwagman/jit-llvm-nvptx/

# Install directory
#INSTALLDIR=${TOPDIR}/install_nojit/${SM}
INSTALLDIR=${TOPDIR}/install/${SM}
if [ "x${OMP}x" == "xyesx" ];
then
 INSTALLDIR=${INSTALLDIR}_omp
fi

LLVM_INSTALL_DIR=${INSTALLDIR}/llvm6-trunk-nvptx
#LLVM_INSTALL_DIR=/home/bjoo/install/llvm-4.0.0-nvptx
# ADD on installed LLVM
export LD_LIBRARY_PATH=${LLVM_INSTALL_DIR}/lib:$LD_LIBRARY_PATH


### ENV VARS for CUDA/MPI
# These are used by the configure script to make make.inc
PK_CUDA_HOME=${CUDA_INSTALL_PATH}
PK_MPI_HOME=${MPI_HOME}
PK_GPU_ARCH=${SM}

### OpenMP
# Open MP enabled
if [ "x${OMP}x" == "xyesx" ]; 
then 
 OMPFLAGS="-fopenmp -D_REENTRANT -L/data/wombat/users/mlwagman/utils/lapack-3.8.0/build/lib -llapack -lblas"
 OMPENABLE="--enable-openmp"
else
 OMPFLAGS="-L/data/wombat/users/mlwagman/utils/lapack-3.8.0/build/lib -llapack -lblas"
 OMPENABLE=""
fi

ARCHFLAGS="-march=x86-64"
DEBUGFLAGS="-g"

PK_CXXFLAGS=${OMPFLAGS}" "${ARCHFLAGS}" "${DEBUGFLAGS}" -O3 -std=c++11 -fexceptions -frtti -fPIC"

PK_CFLAGS=${OMPFLAGS}" "${ARCHFLAGS}" "${DEBUGFLAGS}" -O3 -std=gnu99 -fPIC"

### Make
MAKE="make -j 48"

### MPI and compiler choices

PK_CC=mpicc
PK_CXX=mpicxx
PK_LLVM_CXX=/opt/software/gcc-5.4.0/bin/g++
PK_LLVM_CC=/opt/software/gcc-5.4.0/bin/gcc
PK_LLVM_CFLAGS=" -O3 -std=c99 -fPIC"
PK_LLVM_CXXFLAGS=" -O3 -std=c++11 -fPIC" 
QDPJIT_HOST_ARCH="X86;NVPTX"

module list
