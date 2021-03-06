#!/bin/bash
set -e -x

# Build libvosk
cd /opt
git clone https://github.com/alphacep/vosk-api
cd lid_kaldi/native
CXX=i686-w64-mingw32-g++-posix EXT=dll KALDI_ROOT=/opt/kaldi/kaldi OPENFST_ROOT=/opt/kaldi/local OPENBLAS_ROOT=/opt/kaldi/local make -j $(nproc)

# Copy dependencies
cp /usr/lib/gcc/i686-w64-mingw32/*-posix/libstdc++-6.dll /opt/lid_kaldi/native
cp /usr/lib/gcc/i686-w64-mingw32/*-posix/libgcc_s_sjlj-1.dll /opt/lid_kaldi/native
cp /usr/i686-w64-mingw32/lib/libwinpthread-1.dll /opt/lid_kaldi/native

# Copy dlls to output folder
mkdir -p /io/wheelhouse/win32
cp /opt/lid_kaldi/native/*.dll /io/wheelhouse/win32

# Build wheel and put to the output folder
export LID_SOURCE=/opt/lid_kaldi
export LID_PLATFORM=Windows
export LID_ARCHITECTURE=32bit
python3 -m pip -v wheel /opt/lid_kaldi/python --no-deps -w /io/wheelhouse
