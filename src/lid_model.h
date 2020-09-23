// Copyright 2020 Alpha Cephei Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//       http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef LID_MODEL_H_
#define LID_MODEL_H_

#include "base/kaldi-common.h"
#include "online2/online-feature-pipeline.h"
#include "nnet3/nnet-utils.h"
#include "base/kaldi-common.h"
#include "feat/feature-mfcc.h"
#include "feat/wave-reader.h"
#include "util/common-utils.h"
#include "matrix/kaldi-matrix.h"
#include "ivector/voice-activity-detection.h"
#include "feat/feature-functions.h"
#include "util/common-utils.h"
#include "nnet3/nnet-am-decodable-simple.h"
#include "base/timer.h"
#include "nnet3/nnet-utils.h"
#include "ivector/plda.h"

using namespace kaldi;
//using kaldi::int32;
using namespace kaldi::nnet3;
typedef kaldi::int32 int32;
typedef kaldi::int64 int64;
typedef std::string string;
typedef unordered_map<string, Vector<BaseFloat>*, StringHasher> HashType;

class KaldiRecognizer;

class LidModel {

public:
    LidModel(const char *lid_path);
    void Ref();
    void Unref();

protected:
    friend class KaldiRecognizer;
    ~LidModel() {};

    std::string plda_rxfilename;
    std::string train_ivector_rspecifier;
    std::string test_ivector_rspecifier;
    std::string nnet_rxfilename;
    std::string mean_rxfilename;
    std::string transform_rxfilename;
    std::string num_utts_rspecifier;


    HashType train_ivectors;
    std::map<std::string, int32> num_utts;
    VadEnergyOptions opts;
    PldaConfig plda_config;
    SlidingWindowCmnOptions sliding_opts;
    MfccOptions mfcc_opts;
    NnetSimpleComputationOptions opts_nnet3;

    kaldi::nnet3::Nnet lid_nnet;
    Plda plda;
    Vector<BaseFloat> mean;
    Matrix<BaseFloat> transform;
    MfccOptions lidvector_mfcc_opts;

    int ref_cnt_;
};

#endif /* LID_MODEL_H_ */
