@echo off

@if "%1"=="" (
  echo Error:specify the calculation directory.
  goto end
) else (
  echo "start"
)


echo --------------------------------------------------
echo # start : python script
python map_tif2grid.py config.yml
echo # end : python script

echo --------------------------------------------------
echo # start : set fortran env
set Path=C:\ProgramFiles(x86)\IntelSWTools\compilers_and_libraries\windows\bin\intel64;C:\ProgramFiles(x86)\IntelSWTools\compilers_and_libraries\windows\mpi\intel64\bin;C:\ProgramFiles(x86)\IntelSWTools\compilers_and_libraries\windows\redist\intel64\compiler;C:\ProgramFiles(x86)\IntelSWTools\compilers_and_libraries\windows\redist\intel64_win\compiler;C:\ProgramFiles(x86)\IntelSWTools\compilers_and_libraries\windows\redist\intel64_win\mkl;C:\ProgramFiles(x86)\IntelSWTools\compilers_and_libraries\windows\redist\intel64_win\compiler;C:\ProgramFiles(x86)\CommonFiles\Intel\SharedLibraries\redist\intel64_win\mpirt;C:\ProgramFiles(x86)\CommonFiles\Intel\SharedLibraries\redist\ia32_win\mpirt;C:\ProgramFiles(x86)\CommonFiles\Intel\SharedLibraries\redist\intel64_win\compiler;C:\ProgramFiles(x86)\CommonFiles\Intel\SharedLibraries\redist\ia32_win\compiler;C:\Users\riverlink\iRIC\guis\prepost
: no check compatible
set HDF5_DISABLE_VERSION_CHECK=1
echo # end : set fortran env

echo --------------------------------------------------
echo # start : set OMP env
:@set OMP_STACKSIZE=6M
@set OMP_NUM_THREADS=16
:@set OMP_SCHEDULE=dynamic,1
:@set OMP_SCHEDULE=static,1
:@set KMP_LIBRARY=throughput
:@set KMP_AFFINITY=none
:@set KMP_AFFINITY=none
echo # end : set OMP env

echo --------------------------------------------------d
: go
echo %~dp0
echo %%1 = %1
cd %1
..\nays2dh\nays2dh.exe Case1.cgn
cd ..

:end
