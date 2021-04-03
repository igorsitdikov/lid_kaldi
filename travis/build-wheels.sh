#!/bin/bash
set -e -x

# Build liblid
cd /opt
git clone https://github.com/igorsitdikov/lid_kaldi
cd lid_kaldi/native
KALDI_ROOT=/opt/kaldi OPENFST_ROOT=/opt/kaldi/tools/openfst OPENBLAS_ROOT=/opt/kaldi/tools/OpenBLAS/install make -j $(nproc)

# Copy dlls to output folder
mkdir -p /io/wheelhouse/linux
cp /opt/lid_kaldi/native/*.so /io/wheelhouse/linux

# Build wheel and put to the output folder
mkdir -p /opt/wheelhouse
export LID_SOURCE=/opt/lid_kaldi
/opt/python/cp37*/bin/pip -v wheel /opt/lid_kaldi/python --no-deps -w /opt/wheelhouse

# Fix manylinux
for whl in /opt/wheelhouse/*.whl; do
    cp $whl /io/wheelhouse
    auditwheel repair "$whl" --plat manylinux2010_x86_64 -w /io/wheelhouse
done
