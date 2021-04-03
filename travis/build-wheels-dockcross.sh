#!/bin/bash
set -e -x

# Build so file
cd /opt
git clone https://github.com/igorsitdikov/lid_kaldi
cd /opt/lid_kaldi/native
KALDI_ROOT=/opt/kaldi make -j $(nproc)

# Decide architecture name
export LID_SOURCE=/opt/lid_kaldi
case $CROSS_TRIPLE in
    *armv7-*)
        export LID_ARCHITECTURE=armv7l
        ;;
    *aarch64-*)
        export LID_ARCHITECTURE=aarch64
        ;;
esac

# Copy library to output folder
mkdir -p /io/wheelhouse/linux-$LID_ARCHITECTURE
cp /opt/lid_kaldi/native/*.so /io/wheelhouse/linux-$LID_ARCHITECTURE

# Build wheel
pip3 wheel /opt/lid_kaldi/python --no-deps -w /io/wheelhouse
