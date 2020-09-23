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

#include "kaldi_recognizer.h"
#include "json.h"
#include "fstext/fstext-utils.h"
#include "lat/sausages.h"

using namespace fst;
using namespace kaldi::nnet3;

KaldiRecognizer::KaldiRecognizer(Model *model, float sample_frequency) : model_(model), spk_model_(0), sample_frequency_(sample_frequency) {

    model_->Ref();

    feature_pipeline_ = new kaldi::OnlineNnet2FeaturePipeline (model_->feature_info_);
    silence_weighting_ = new kaldi::OnlineSilenceWeighting(*model_->trans_model_, model_->feature_info_.silence_weighting_config, 3);

    g_fst_ = NULL;
    decode_fst_ = NULL;

    if (!model_->hclg_fst_) {
        if (model_->hcl_fst_ && model_->g_fst_) {
            decode_fst_ = LookaheadComposeFst(*model_->hcl_fst_, *model_->g_fst_, model_->disambig_);
        } else {
            KALDI_ERR << "Can't create decoding graph";
        }
    }

    decoder_ = new kaldi::SingleUtteranceNnet3Decoder(model_->nnet3_decoding_config_,
            *model_->trans_model_,
            *model_->decodable_info_,
            model_->hclg_fst_ ? *model_->hclg_fst_ : *decode_fst_,
            feature_pipeline_);

    frame_offset_ = 0;
    input_finalized_ = false;
    spk_feature_ = NULL;

    InitRescoring();
}

KaldiRecognizer::KaldiRecognizer(Model *model, float sample_frequency, char const *grammar) : model_(model), spk_model_(0), sample_frequency_(sample_frequency)
{
    model_->Ref();

    feature_pipeline_ = new kaldi::OnlineNnet2FeaturePipeline (model_->feature_info_);
    silence_weighting_ = new kaldi::OnlineSilenceWeighting(*model_->trans_model_, model_->feature_info_.silence_weighting_config, 3);

    g_fst_ = new StdVectorFst();
    if (model_->hcl_fst_) {
        g_fst_->AddState();
        g_fst_->SetStart(0);
        g_fst_->AddState();
        g_fst_->SetFinal(1, fst::TropicalWeight::One());
        g_fst_->AddArc(1, StdArc(0, 0, fst::TropicalWeight::One(), 0));

        // Create simple word loop FST
        stringstream ss(grammar);
        string token;

        while (getline(ss, token, ' ')) {
            int32 id = model_->word_syms_->Find(token);
            g_fst_->AddArc(0, StdArc(id, id, fst::TropicalWeight::One(), 1));
        }
        ArcSort(g_fst_, ILabelCompare<StdArc>());

        decode_fst_ = LookaheadComposeFst(*model_->hcl_fst_, *g_fst_, model_->disambig_);
    } else {
        decode_fst_ = NULL;
        KALDI_ERR << "Can't create decoding graph";
    }

    decoder_ = new kaldi::SingleUtteranceNnet3Decoder(model_->nnet3_decoding_config_,
            *model_->trans_model_,
            *model_->decodable_info_,
            model_->hclg_fst_ ? *model_->hclg_fst_ : *decode_fst_,
            feature_pipeline_);

    frame_offset_ = 0;
    input_finalized_ = false;
    spk_feature_ = NULL;

    InitRescoring();
}

KaldiRecognizer::KaldiRecognizer(Model *model, SpkModel *spk_model, float sample_frequency) : model_(model), spk_model_(spk_model), sample_frequency_(sample_frequency) {

    model_->Ref();
    spk_model->Ref();

    feature_pipeline_ = new kaldi::OnlineNnet2FeaturePipeline (model_->feature_info_);
    silence_weighting_ = new kaldi::OnlineSilenceWeighting(*model_->trans_model_, model_->feature_info_.silence_weighting_config, 3);

    decode_fst_ = NULL;
    g_fst_ = NULL;

    if (!model_->hclg_fst_) {
        if (model_->hcl_fst_ && model_->g_fst_) {
            decode_fst_ = LookaheadComposeFst(*model_->hcl_fst_, *model_->g_fst_, model_->disambig_);
        } else {
            KALDI_ERR << "Can't create decoding graph";
        }
    }

    decoder_ = new kaldi::SingleUtteranceNnet3Decoder(model_->nnet3_decoding_config_,
            *model_->trans_model_,
            *model_->decodable_info_,
            model_->hclg_fst_ ? *model_->hclg_fst_ : *decode_fst_,
            feature_pipeline_);

    frame_offset_ = 0;
    input_finalized_ = false;

    spk_feature_ = new OnlineMfcc(spk_model_->spkvector_mfcc_opts);

    InitRescoring();
}

// Computes an xvector from a chunk of speech features.
static void RunNnetComputation(const MatrixBase<BaseFloat> &features,
                               const nnet3::Nnet &nnet, nnet3::CachingOptimizingCompiler *compiler,
                               Vector<BaseFloat> *xvector)
{
    nnet3::ComputationRequest request;
    request.need_model_derivative = false;
    request.store_component_stats = false;
    request.inputs.push_back(
            nnet3::IoSpecification("input", 0, features.NumRows()));
    nnet3::IoSpecification output_spec;
    output_spec.name = "output";
    output_spec.has_deriv = false;
    output_spec.indexes.resize(1);
    request.outputs.resize(1);
    request.outputs[0].Swap(&output_spec);
    shared_ptr<const nnet3::NnetComputation> computation = compiler->Compile(request);
    nnet3::Nnet *nnet_to_update = NULL;  // we're not doing any update.
    nnet3::NnetComputer computer(nnet3::NnetComputeOptions(), *computation,
                                 nnet, nnet_to_update);
    CuMatrix<BaseFloat> input_feats_cu(features);
    computer.AcceptInput("input", &input_feats_cu);
    computer.Run();
    CuMatrix<BaseFloat> cu_output;
    computer.GetOutputDestructive("output", &cu_output);
    xvector->Resize(cu_output.NumCols());
    xvector->CopyFromVec(cu_output.Row(0));
}
KaldiRecognizer::KaldiRecognizer(LidModel *lid_model, float sample_frequency) : lid_model_(lid_model), sample_frequency_(sample_frequency) {
    lid_model_->Ref();
    lid_feature_ = new OnlineMfcc(lid_model_->mfcc_opts);
}
KaldiRecognizer::~KaldiRecognizer() {
    delete feature_pipeline_;
    delete silence_weighting_;
    delete decoder_;
    delete g_fst_;
    delete decode_fst_;
    delete spk_feature_;
    delete lm_fst_;
    delete lid_feature_;

    if (model_)
         model_->Unref();
    if (spk_model_)
         spk_model_->Unref();
    if (lid_model_)
         lid_model_->Unref();
}

void KaldiRecognizer::Calculate(const char *data, int len) {
//    Mfcc mfcc(lid_model_->mfcc_opts);
    Vector<BaseFloat> wave;
    wave.Resize(len / 2, kUndefined);
    for (int i = 0; i < len / 2; i++)
        wave(i) = *(((short *)data) + i);
//    BaseFloat vtln_warp = 1.0;
    Plda plda(lid_model_ -> plda);
//    Matrix<BaseFloat> features;
//    mfcc.ComputeFeatures(wave, sample_frequency_,
//                         vtln_warp, &features);
//    *****************************************************
    frame_offset_ = 0;

    lid_feature_->AcceptWaveform(sample_frequency_, wave);
    KALDI_LOG << "lid_feature_ " << lid_feature_;
    int num_frames = lid_feature_->NumFramesReady() - frame_offset_ * 3;
    Matrix<BaseFloat> features(num_frames, lid_feature_->Dim());

    for (int i = 0; i < num_frames; ++i) {
        Vector<BaseFloat> feat(lid_feature_->Dim());
        lid_feature_->GetFrame(i + frame_offset_ * 3, &feat);
        features.CopyRowFromVec(feat, i);
    }

    //    *****************************************************
    int32 compression_method_in = 1;
    CompressionMethod compression_method = static_cast<CompressionMethod>(
            compression_method_in);
    Matrix<BaseFloat> compressedMatrix;
    const CompressedMatrix &matrix = CompressedMatrix(features, compression_method);
    compressedMatrix.Resize(matrix.NumRows(), matrix.NumCols());
    matrix.CopyToMat(&compressedMatrix, kNoTrans);

    Matrix<BaseFloat> cmvn_feat(compressedMatrix.NumRows(),
                                compressedMatrix.NumCols(), kUndefined);

    SlidingWindowCmn(lid_model_ -> sliding_opts, compressedMatrix, &cmvn_feat);

    int32 num_done = 0, num_err = 0;
    int32 num_unvoiced = 0;
    double tot_length = 0.0, tot_decision = 0.0;
    bool omit_unvoiced_utts = false;

    Vector<BaseFloat> vad_result(compressedMatrix.NumRows());

    ComputeVadEnergy(lid_model_ -> opts, compressedMatrix, &vad_result);

    double sum = vad_result.Sum();
    if (sum == 0.0) {
        KALDI_WARN << "No frames were judged voiced for utterance default";
        num_unvoiced++;
    } else {
        num_done++;
    }
    tot_decision += vad_result.Sum();
    tot_length += vad_result.Dim();

    const Vector<BaseFloat> &voiced = vad_result;

    if (cmvn_feat.NumRows() != voiced.Dim()) {
        KALDI_WARN << "Mismatch in number for frames " << cmvn_feat.NumRows()
                   << " for features and VAD " << voiced.Dim()
                   << ", for utterance default " ;
        num_err++;
    }
    if (voiced.Sum() == 0.0) {
        KALDI_WARN << "No features were judged as voiced for utterance default";
        num_err++;
    }
    int32 dim = 0;
    for (int32 i = 0; i < voiced.Dim(); i++)
        if (voiced(i) != 0.0)
            dim++;
    Matrix<BaseFloat> voiced_feat(dim, cmvn_feat.NumCols());
    int32 index = 0;
    for (int32 i = 0; i < cmvn_feat.NumRows(); i++) {
        if (voiced(i) != 0.0) {
            KALDI_ASSERT(voiced(i) == 1.0); // should be zero or one.
            voiced_feat.Row(index).CopyFromVec(cmvn_feat.Row(i));
            index++;
        }
    }
    KALDI_ASSERT(index == dim);

    std::string use_gpu = "no";
    std::string cached_compiler_in;
    std::string cached_compiler_out;
    int32 chunk_size = 10000,
            min_chunk_size = 25;
    bool pad_input = true;
    CachingOptimizingCompilerOptions compiler_config;
    compiler_config.cache_capacity = 64;
    lid_model_ -> opts_nnet3.acoustic_scale = 1.0;
    CachingOptimizingCompiler compiler( lid_model_ -> lid_nnet, lid_model_ -> opts_nnet3.optimize_config, compiler_config);

    int32 xvector_dim = lid_model_-> lid_nnet.OutputDim("output");
    Vector<BaseFloat> xvector_result;
    int32 num_rows = voiced_feat.NumRows(),
            feat_dim = voiced_feat.NumCols(),
            this_chunk_size = chunk_size;
    if (!pad_input && num_rows < min_chunk_size) {
        KALDI_WARN << "Minimum chunk size of " << min_chunk_size
                   << " is greater than the number of rows "
                   << "in utterance: default";
    } else if (num_rows < chunk_size) {
        KALDI_LOG << "Chunk size of " << chunk_size << " is greater than "
                  << "the number of rows in utterance: default"
                  << ", using chunk size  of " << num_rows;
        this_chunk_size = num_rows;
    } else if (chunk_size == -1) {
        this_chunk_size = num_rows;
    }

    int32 num_chunks = ceil(
            num_rows / static_cast<BaseFloat>(this_chunk_size));
    Vector<BaseFloat> xvector_avg(xvector_dim, kSetZero);
    BaseFloat tot_weight = 0.0;
    int32 num_utts = 0;
    int64 frame_count = 0;
    // Iterate over the feature chunks.
    for (int32 chunk_indx = 0; chunk_indx < num_chunks; chunk_indx++) {
        // If we're nearing the end of the input, we may need to shift the
        // offset back so that we can get this_chunk_size frames of input to
        // the nnet.
        int32 offset = std::min(
                this_chunk_size, num_rows - chunk_indx * this_chunk_size);
        if (!pad_input && offset < min_chunk_size)
            continue;
        SubMatrix<BaseFloat> sub_features(
                voiced_feat, chunk_indx * this_chunk_size, offset, 0, feat_dim);
        Vector<BaseFloat> xvector;
        tot_weight += offset;

        // Pad input if the offset is less than the minimum chunk size
        if (pad_input && offset < min_chunk_size) {
            Matrix<BaseFloat> padded_features(min_chunk_size, feat_dim);
            int32 left_context = (min_chunk_size - offset) / 2;
            int32 right_context = min_chunk_size - offset - left_context;
            for (int32 i = 0; i < left_context; i++) {
                padded_features.Row(i).CopyFromVec(sub_features.Row(0));
            }
            for (int32 i = 0; i < right_context; i++) {
                padded_features.Row(min_chunk_size - i - 1).CopyFromVec(sub_features.Row(offset - 1));
            }
            padded_features.Range(left_context, offset, 0, feat_dim).CopyFromMat(sub_features);
            RunNnetComputation(padded_features, lid_model_->lid_nnet, &compiler, &xvector);
        } else {
            RunNnetComputation(sub_features, lid_model_->lid_nnet, &compiler, &xvector);
        }
        xvector_result = xvector;
        xvector_avg.AddVec(offset, xvector);
    }

    xvector_avg.Scale(1.0 / tot_weight);

    frame_count += voiced_feat.NumRows();

    if (num_utts % 10 == 0)
        KALDI_LOG << "Processed " << num_utts << " utterances";
    KALDI_VLOG(2) << "Processed features for key default";

    int64 num_train_ivectors = 0, num_train_errs = 0, num_test_ivectors = 0;
    double tot_test_renorm_scale = 0.0, tot_train_renorm_scale = 0.0;
    HashType test_ivectors;
    KALDI_LOG << "Reading test iVectors";
    std::string utt = "default";
    if (test_ivectors.count(utt) != 0) {
        KALDI_ERR << "Duplicate test iVector found for utterance " << utt;
    }

    xvector_result.AddVec(-1.0, lid_model_ -> mean);
    const Vector<BaseFloat> &vec(xvector_result);
    int32 transform_rows = lid_model_ -> transform.NumRows();
    int32 transform_cols = lid_model_ -> transform.NumCols();
    int32 vec_dim = vec.Dim();
    Vector<BaseFloat> vec_out(transform_rows);
    if (transform_cols == vec_dim) {
        vec_out.AddMatVec(1.0,  lid_model_ -> transform, kNoTrans, vec, 0.0);
    } else {
        if (transform_cols != vec_dim + 1) {
            KALDI_ERR << "Dimension mismatch: input vector has dimension "
                      << vec.Dim() << " and transform has " << transform_cols
                      << " columns.";
        }
        vec_out.CopyColFromMat(lid_model_ -> transform, vec_dim);
        vec_out.AddMatVec(1.0, lid_model_ -> transform.Range(0, lid_model_ -> transform.NumRows(),
                                                            0, vec_dim), kNoTrans, vec, 1.0);
    }
    int32 num_examples = 1; // this value is always used for test (affects the
    int32 plda_dim = plda.Dim();
    Vector<BaseFloat> *transformed_ivector = new Vector<BaseFloat>(plda_dim);
    tot_test_renorm_scale += plda.TransformIvector(lid_model_ -> plda_config, vec_out,
                                                                num_examples,
                                                                transformed_ivector);
    test_ivectors[utt] = transformed_ivector;
    bool binary = false;

    double sums = 0.0, sumsq = 0.0;
    typedef unordered_map<string, Vector<BaseFloat>*, StringHasher> HashType;
    std::map<std::string, int32> langs = lid_model_ -> num_utts;
    for (auto const& x : langs) {
        std::string key1 = x.first;
        if (lid_model_ -> train_ivectors.count(key1) == 0) {
            KALDI_WARN << "Key " << key1 << " not present in training iVectors.";
            continue;
        }
        if (test_ivectors.count(utt) == 0) {
            KALDI_WARN << "Key " << utt << " not present in test iVectors.";
            continue;
        }
        const Vector<BaseFloat> *train_ivector = lid_model_ -> train_ivectors[key1];
        const Vector<BaseFloat> *test_ivector = test_ivectors[utt];

        Vector<double> train_ivector_dbl(*train_ivector),
                test_ivector_dbl(*test_ivector);

        int32 num_train_examples;
        num_train_examples = lid_model_ -> num_utts[key1];

        BaseFloat score = plda.LogLikelihoodRatio(train_ivector_dbl,
                                                               num_train_examples,
                                                               test_ivector_dbl);
        sums += score;
        sumsq += score * score;

        scores_.insert(std::pair<std::string, BaseFloat>(key1, score));
    }

    for (HashType::iterator iter = lid_model_ -> train_ivectors.begin();
         iter != lid_model_ -> train_ivectors.end(); ++iter)
        delete iter->second;
    for (HashType::iterator iter = test_ivectors.begin();
         iter != test_ivectors.end(); ++iter)
        delete iter->second;
}

void KaldiRecognizer::InitRescoring()
{
    if (model_->std_lm_fst_) {
        fst::CacheOptions cache_opts(true, 50000);
        fst::MapFstOptions mapfst_opts(cache_opts);
        fst::StdToLatticeMapper<kaldi::BaseFloat> mapper;
        lm_fst_ = new fst::MapFst<fst::StdArc, kaldi::LatticeArc, fst::StdToLatticeMapper<kaldi::BaseFloat> >(*model_->std_lm_fst_, mapper, mapfst_opts);
    } else {
        lm_fst_ = NULL;
    }
}

void KaldiRecognizer::CleanUp()
{
    delete silence_weighting_;
    silence_weighting_ = new kaldi::OnlineSilenceWeighting(*model_->trans_model_, model_->feature_info_.silence_weighting_config, 3);

    frame_offset_ += decoder_->NumFramesDecoded();
    decoder_->InitDecoding(frame_offset_);
}

void KaldiRecognizer::UpdateSilenceWeights()
{
    if (silence_weighting_->Active() && feature_pipeline_->NumFramesReady() > 0 &&
        feature_pipeline_->IvectorFeature() != NULL) {
        vector<pair<int32, BaseFloat> > delta_weights;
        silence_weighting_->ComputeCurrentTraceback(decoder_->Decoder());
        silence_weighting_->GetDeltaWeights(feature_pipeline_->NumFramesReady(),
                                          frame_offset_ * 3,
                                          &delta_weights);
        feature_pipeline_->UpdateFrameWeights(delta_weights);
    }
}

bool KaldiRecognizer::AcceptWaveform(const char *data, int len)
{
    Vector<BaseFloat> wave;
    wave.Resize(len / 2, kUndefined);
    for (int i = 0; i < len / 2; i++)
        wave(i) = *(((short *)data) + i);
    return AcceptWaveform(wave);
}

bool KaldiRecognizer::AcceptWaveform(const short *sdata, int len)
{
    Vector<BaseFloat> wave;
    wave.Resize(len, kUndefined);
    for (int i = 0; i < len; i++)
        wave(i) = sdata[i];
    return AcceptWaveform(wave);
}

bool KaldiRecognizer::AcceptWaveform(const float *fdata, int len)
{
    Vector<BaseFloat> wave;
    wave.Resize(len, kUndefined);
    for (int i = 0; i < len; i++)
        wave(i) = fdata[i];
    return AcceptWaveform(wave);
}

bool KaldiRecognizer::AcceptWaveform(Vector<BaseFloat> &wdata)
{
    if (input_finalized_) {
        CleanUp();
        input_finalized_ = false;
    }

    feature_pipeline_->AcceptWaveform(sample_frequency_, wdata);
    KALDI_LOG << "feature_pipeline_ " << feature_pipeline_;
    UpdateSilenceWeights();
    decoder_->AdvanceDecoding();

    if (spk_feature_) {
        spk_feature_->AcceptWaveform(sample_frequency_, wdata);
        KALDI_LOG << "spk_feature_ " << spk_feature_;
    }

    if (decoder_->EndpointDetected(model_->endpoint_config_)) {
        return true;
    }

    return false;
}


void KaldiRecognizer::GetSpkVector(Vector<BaseFloat> &xvector)
{
    int num_frames = spk_feature_->NumFramesReady() - frame_offset_ * 3;
    Matrix<BaseFloat> mfcc(num_frames, spk_feature_->Dim());

    for (int i = 0; i < num_frames; ++i) {
       Vector<BaseFloat> feat(spk_feature_->Dim());
//       KALDI_LOG << "feat before " << feat;
       spk_feature_->GetFrame(i + frame_offset_ * 3, &feat);
//       KALDI_LOG << "feat after " << feat;
       mfcc.CopyRowFromVec(feat, i);
    }

    KALDI_LOG << "mfcc " << mfcc;
    SlidingWindowCmnOptions cmvn_opts;
    Matrix<BaseFloat> features(mfcc.NumRows(), mfcc.NumCols(), kUndefined);
    cmvn_opts.cmn_window = 300;
    cmvn_opts.center = true;

    SlidingWindowCmn(cmvn_opts, mfcc, &features);

    nnet3::NnetSimpleComputationOptions opts;
    nnet3::CachingOptimizingCompilerOptions compiler_config;
    nnet3::CachingOptimizingCompiler compiler(spk_model_->speaker_nnet, opts.optimize_config, compiler_config);

    KALDI_LOG << "features " << features;
    RunNnetComputation(features, spk_model_->speaker_nnet, &compiler, &xvector);
}

const char* KaldiRecognizer::Result()
{

    if (!input_finalized_) {
        decoder_->FinalizeDecoding();
        input_finalized_ = true;
    }

    if (decoder_->NumFramesDecoded() == 0) {
        last_result_ = "[]";
        return last_result_.c_str();
    }

    kaldi::CompactLattice clat;
    decoder_->GetLattice(true, &clat);

    if (model_->std_lm_fst_) {
        Lattice lat1;

        ConvertLattice(clat, &lat1);
        fst::ScaleLattice(fst::GraphLatticeScale(-1.0), &lat1);
        fst::ArcSort(&lat1, fst::OLabelCompare<kaldi::LatticeArc>());
        kaldi::Lattice composed_lat;
        fst::Compose(lat1, *lm_fst_, &composed_lat);
        fst::Invert(&composed_lat);
        kaldi::CompactLattice determinized_lat;
        DeterminizeLattice(composed_lat, &determinized_lat);
        fst::ScaleLattice(fst::GraphLatticeScale(-1), &determinized_lat);
        fst::ArcSort(&determinized_lat, fst::OLabelCompare<kaldi::CompactLatticeArc>());

        kaldi::ConstArpaLmDeterministicFst const_arpa_fst(model_->const_arpa_);
        kaldi::CompactLattice composed_clat;
        kaldi::ComposeCompactLatticeDeterministic(determinized_lat, &const_arpa_fst, &composed_clat);
        kaldi::Lattice composed_lat1;
        ConvertLattice(composed_clat, &composed_lat1);
        fst::Invert(&composed_lat1);
        DeterminizeLattice(composed_lat1, &clat);
    }

    fst::ScaleLattice(fst::LatticeScale(9.0, 10.0), &clat);
    CompactLattice aligned_lat;
    if (model_->winfo_) {
        WordAlignLattice(clat, *model_->trans_model_, *model_->winfo_, 0, &aligned_lat);
    } else {
        aligned_lat = clat;
    }

    MinimumBayesRisk mbr(aligned_lat);
    const vector<BaseFloat> &conf = mbr.GetOneBestConfidences();
    const vector<int32> &words = mbr.GetOneBest();
    const vector<pair<BaseFloat, BaseFloat> > &times =
          mbr.GetOneBestTimes();

    int size = words.size();

    json::JSON obj;
    stringstream text;
//
    // Create JSON object
    for (int i = 0; i < size; i++) {
        json::JSON word;
        word["data"]["note"] = model_->word_syms_->Find(words[i]);
        word["start"] = (frame_offset_ + times[i].first) * 0.03;
        word["end"] = (frame_offset_ + times[i].second) * 0.03;
        word["conf"] = conf[i];
        word["drag"] = false;
        word["resize"] = false;
        obj.append(word);

//        if (i) {
//            text << " ";
//        }
//        text << model_->word_syms_->Find(words[i]);
    }
//    obj["text"] = text.str();

    if (spk_model_) {
        Vector<BaseFloat> xvector;
        GetSpkVector(xvector);
        KALDI_LOG << "xvector " << xvector;
        for (int i = 0; i < xvector.Dim(); i++) {
            obj["spk"].append(xvector(i));
        }
    }

    last_result_ = obj.dump();
    return last_result_.c_str();
}
const char* KaldiRecognizer::LangResult(const char *data, int len) {

    Calculate(data, len);

    json::JSON obj;
//    for (auto i = scores_.begin(); i != scores_.end(); ++i) {
    for (auto const& x : scores_) {
        json::JSON res;
        res["language"] = x.first;
//        res["obj"]["key"] = x.first;
        res["score"] = x.second;
//        res["obj"]["value"] = x.second;
        obj.append(res);
    }

    scores_.clear();
    lang_result_ = obj.dump();
    lid_model_->Unref();
    delete lid_model_;
    return lang_result_.c_str();
//    if (!input_finalized_) {
//        feature_pipeline_->InputFinished();
//        UpdateSilenceWeights();
//        decoder_->AdvanceDecoding();
//        decoder_->FinalizeDecoding();
//        input_finalized_ = true;
//        return Result();
//    } else {
//        last_result_ = "[]";
//        return last_result_.c_str();
//    }
}
const char* KaldiRecognizer::PartialResult()
{
    json::JSON res;
    if (decoder_->NumFramesDecoded() == 0) {
        res["partial"] = "";
        last_result_ = res.dump();
        return last_result_.c_str();
    }

    kaldi::Lattice lat;
    decoder_->GetBestPath(false, &lat);
    vector<kaldi::int32> alignment, words;
    LatticeWeight weight;
    GetLinearSymbolSequence(lat, &alignment, &words, &weight);

    ostringstream text;
    for (size_t i = 0; i < words.size(); i++) {
        if (i) {
            text << " ";
        }
        text << model_->word_syms_->Find(words[i]);
    }
    res["partial"] = text.str();

    last_result_ = res.dump();
    return last_result_.c_str();
}

const char* KaldiRecognizer::FinalResult()
{
    if (!input_finalized_) {
        feature_pipeline_->InputFinished();
        UpdateSilenceWeights();
        decoder_->AdvanceDecoding();
        decoder_->FinalizeDecoding();
        input_finalized_ = true;
        return Result();
    } else {
        last_result_ = "[]";
        return last_result_.c_str();
    }
}

