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

#include "lid_model.h"

LidModel::LidModel(const char *lid_path) {
    std::string language_path_str(lid_path);

    kaldi::ParseOptions po("something");
//    kaldi::ParseOptions po_vad("vad");
//    plda_config.Register(&po);
//    opts.Register(&po_vad);
//    mfcc_opts.Register(&po);
//    po.ReadConfigFile(language_path_str + "/conf/model.conf");

    ReadConfigFromFile(language_path_str + "/mfcc.conf", &mfcc_opts);
    ReadConfigFromFile(language_path_str + "/vad.conf", &opts);
    lidvector_mfcc_opts.frame_opts.allow_downsample = true; // It is safe to downsample
    plda_rxfilename = language_path_str + "/plda_adapt.smooth0.1";
    mean_rxfilename = language_path_str + "/mean.vec";
    transform_rxfilename = language_path_str + "/transform.mat";
    nnet_rxfilename = language_path_str + "/final.ext.raw";
    train_ivector_rspecifier = "ark:" + language_path_str + "/xvector.final.train.scp";
    num_utts_rspecifier = "ark:" + language_path_str + "/num_utts.ark";

    sliding_opts.cmn_window = 300;
    sliding_opts.center = true;

    RandomAccessInt32Reader num_utts_reader(num_utts_rspecifier);

    ReadKaldiObject(plda_rxfilename, &plda);

    ReadKaldiObject(mean_rxfilename, &mean);

    ReadKaldiObject(transform_rxfilename, &transform);

    ReadKaldiObject(nnet_rxfilename, &lid_nnet);

    double tot_test_renorm_scale = 0.0, tot_train_renorm_scale = 0.0;
    int64 num_train_ivectors = 0, num_train_errs = 0, num_test_ivectors = 0;
    int32 dim = plda.Dim();
    SequentialBaseFloatVectorReader train_ivector_reader(train_ivector_rspecifier);
    for (; !train_ivector_reader.Done(); train_ivector_reader.Next()) {
        std::string spk = train_ivector_reader.Key();
        if (train_ivectors.count(spk) != 0) {
            KALDI_ERR << "Duplicate training iVector found for speaker " << spk;
        }
        const Vector<BaseFloat> &ivector = train_ivector_reader.Value();
        int32 num_examples;
        if (!num_utts_rspecifier.empty()) {
            if (!num_utts_reader.HasKey(spk)) {
                KALDI_WARN << "Number of utterances not given for speaker " << spk;
                num_train_errs++;
                continue;
            }
            num_examples = num_utts_reader.Value(spk);
        } else {
            num_examples = 1;
        }
        num_utts.insert(std::pair<std::string, int32>(spk, num_examples));

        Vector<BaseFloat> *transformed_ivector = new Vector<BaseFloat>(dim);
        tot_train_renorm_scale += plda.TransformIvector(plda_config, ivector,
                                                        num_examples,
                                                        transformed_ivector);
        train_ivectors[spk] = transformed_ivector;
        num_train_ivectors++;
    }

    KALDI_LOG << "Read " << num_train_ivectors << " training iVectors, "
              << "errors on " << num_train_errs;
    SetBatchnormTestMode(true, &lid_nnet);
    SetDropoutTestMode(true, &lid_nnet);
    CollapseModel(nnet3::CollapseModelConfig(), &lid_nnet);

    ref_cnt_ = 1;
}

void LidModel::Ref()
{
    ref_cnt_++;
}

void LidModel::Unref()
{
    ref_cnt_--;
    if (ref_cnt_ == 0) {
        delete this;
    }
}
