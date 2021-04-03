#!/bin/bash

set -e -x 
#docker build --file Dockerfile.win --tag alphacep/kaldi-win:latest .
docker run --rm -v /home/codex/Documents/Projects/trash/asr/lid_kaldi:/io alphacep/kaldi-win /io/travis/build-wheels-win.sh
