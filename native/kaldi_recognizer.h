// Copyright 2019 Alpha Cephei Inc.
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

#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "fstext/fstext-lib.h"
#include "fstext/fstext-utils.h"
#include "decoder/lattice-faster-decoder.h"
#include "feat/feature-mfcc.h"
#include "lat/kaldi-lattice.h"
#include "lat/word-align-lattice.h"
#include "nnet3/am-nnet-simple.h"
#include "nnet3/nnet-am-decodable-simple.h"
#include "nnet3/nnet-utils.h"

#include "lid_model.h"

using namespace kaldi;

class KaldiRecognizer {
    public:
        KaldiRecognizer(LidModel *lid_model, float sample_frequency);
        ~KaldiRecognizer();
        const char* LangResult();
        void AcceptWaveform(const char *data, int len);
        void AcceptWaveform(const short *sdata, int len);
        void AcceptWaveform(const float *fdata, int len);

    private:
        void PldaScoring();
        void Nnet3XvectorCompute(Matrix <BaseFloat> voiced_feat);
        int Calculate();
        LidModel *lid_model_;
        OnlineBaseFeature *lid_feature_;
        std::string GetLanguage(std::string lg);
        void AcceptWaveform(Vector<BaseFloat> &wdata);
        std::map<std::string, BaseFloat> scores_;
        float sample_frequency_;
        int32 frame_offset_;
        string lang_result_;
        Vector <BaseFloat> voiced;
        Vector <BaseFloat> xvector_result;
};
