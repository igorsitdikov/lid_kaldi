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

// Computes an xvector from a chunk of speech features.
static void RunNnetComputation(const MatrixBase <BaseFloat> &features,
                               const nnet3::Nnet &nnet, nnet3::CachingOptimizingCompiler *compiler,
                               Vector <BaseFloat> *xvector) {
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
    std::shared_ptr<const nnet3::NnetComputation> computation = compiler->Compile(request);
    nnet3::Nnet *nnet_to_update = NULL;  // we're not doing any update.
    nnet3::NnetComputer computer(nnet3::NnetComputeOptions(), *computation,
                                 nnet, nnet_to_update);
    CuMatrix <BaseFloat> input_feats_cu(features);
    computer.AcceptInput("input", &input_feats_cu);
    computer.Run();
    CuMatrix <BaseFloat> cu_output;
    computer.GetOutputDestructive("output", &cu_output);
    xvector->Resize(cu_output.NumCols());
    xvector->CopyFromVec(cu_output.Row(0));
}

KaldiRecognizer::KaldiRecognizer(LidModel *lid_model, float sample_frequency) : lid_model_(lid_model),
                                                                                sample_frequency_(sample_frequency) {
    lid_model_->Ref();
    lid_feature_ = new OnlineMfcc(lid_model_->mfcc_opts);
}

KaldiRecognizer::~KaldiRecognizer() {

    if (lid_model_) {
        delete lid_feature_;
    }
}

#define MIN_LANG_FEATS 50

void KaldiRecognizer::PldaScoring() {

    int64 num_train_ivectors = 0, num_train_errs = 0, num_test_ivectors = 0;
    double tot_test_renorm_scale = 0.0, tot_train_renorm_scale = 0.0;
    HashType test_ivectors;
    KALDI_LOG << "Reading test iVectors";
    std::string utt = "default";
    if (test_ivectors.count(utt) != 0) {
        KALDI_ERR << "Duplicate test iVector found for utterance " << utt;
    }

    xvector_result.AddVec(-1.0, lid_model_->mean);
    const Vector <BaseFloat> &vec(xvector_result);
    int32 transform_rows = lid_model_->transform.NumRows();
    int32 transform_cols = lid_model_->transform.NumCols();
    int32 vec_dim = vec.Dim();
    Vector <BaseFloat> vec_out(transform_rows);
    if (transform_cols == vec_dim) {
        vec_out.AddMatVec(1.0, lid_model_->transform, kNoTrans, vec, 0.0);
    } else {
        if (transform_cols != vec_dim + 1) {
            KALDI_ERR << "Dimension mismatch: input vector has dimension "
                      << vec.Dim() << " and transform has " << transform_cols
                      << " columns.";
        }
        vec_out.CopyColFromMat(lid_model_->transform, vec_dim);
        vec_out.AddMatVec(1.0, lid_model_->transform.Range(0, lid_model_->transform.NumRows(),
                                                           0, vec_dim), kNoTrans, vec, 1.0);
    }

    int32 num_examples = 1;   // this value is always used for test (affects the
                           // length normalization in the TransformIvector
                           // function).
    Plda plda(lid_model_->plda);

    int32 plda_dim = plda.Dim();
    Vector <BaseFloat> *transformed_ivector = new Vector<BaseFloat>(plda_dim);
    tot_test_renorm_scale += plda.TransformIvector(lid_model_->plda_config, vec_out,
                                                   num_examples,
                                                   transformed_ivector);
    test_ivectors[utt] = transformed_ivector;
    bool binary = false;

    double sums = 0.0, sumsq = 0.0;
    typedef unordered_map<string, Vector < BaseFloat>*, StringHasher > HashType;
    std::map <std::string, int32> langs = lid_model_->num_utts;
    for (auto const &x : langs) {
        std::string key1 = x.first;
        if (lid_model_->train_ivectors.count(key1) == 0) {
            KALDI_WARN << "Key " << key1 << " not present in training iVectors.";
            continue;
        }
        if (test_ivectors.count(utt) == 0) {
            KALDI_WARN << "Key " << utt << " not present in test iVectors.";
            continue;
        }
        const Vector <BaseFloat> *train_ivector = lid_model_->train_ivectors[key1];
        const Vector <BaseFloat> *test_ivector = test_ivectors[utt];

        Vector<double> train_ivector_dbl(*train_ivector),
                test_ivector_dbl(*test_ivector);

        int32 num_train_examples;
        num_train_examples = lid_model_->num_utts[key1];

        BaseFloat score = plda.LogLikelihoodRatio(train_ivector_dbl,
                                                  num_train_examples,
                                                  test_ivector_dbl);
        sums += score;
        sumsq += score * score;

        scores_.insert(std::pair<std::string, BaseFloat>(key1, score));
    }

//    for (HashType::iterator iter = lid_model_->train_ivectors.begin();
//         iter != lid_model_->train_ivectors.end(); ++iter)
//        delete iter->second;
    for (HashType::iterator iter = test_ivectors.begin();
         iter != test_ivectors.end(); ++iter)
        delete iter->second;
}

void KaldiRecognizer::Nnet3XvectorCompute(Matrix <BaseFloat> voiced_feat) {

    std::string use_gpu = "no";
    std::string cached_compiler_in;
    std::string cached_compiler_out;
    int32 chunk_size = 10000,
            min_chunk_size = 25;
    bool pad_input = true;
    CachingOptimizingCompilerOptions compiler_config;
    compiler_config.cache_capacity = 64;
    lid_model_->opts_nnet3.acoustic_scale = 1.0;
    CachingOptimizingCompiler compiler(lid_model_->lid_nnet, lid_model_->opts_nnet3.optimize_config, compiler_config);

    int32 xvector_dim = lid_model_->lid_nnet.OutputDim("output");

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
    Vector <BaseFloat> xvector_avg(xvector_dim, kSetZero);
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
        SubMatrix <BaseFloat> sub_features(
                voiced_feat, chunk_indx * this_chunk_size, offset, 0, feat_dim);
        Vector <BaseFloat> xvector;
        tot_weight += offset;

        // Pad input if the offset is less than the minimum chunk size
        if (pad_input && offset < min_chunk_size) {
            Matrix <BaseFloat> padded_features(min_chunk_size, feat_dim);
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
}

void KaldiRecognizer::AcceptWaveform(const char *data, int len)
{
    Vector<BaseFloat> wave;
    wave.Resize(len / 2, kUndefined);
    for (int i = 0; i < len / 2; i++)
        wave(i) = *(((short *)data) + i);
    AcceptWaveform(wave);
}

void KaldiRecognizer::AcceptWaveform(const short *sdata, int len)
{
    Vector<BaseFloat> wave;
    wave.Resize(len, kUndefined);
    for (int i = 0; i < len; i++)
        wave(i) = sdata[i];
    AcceptWaveform(wave);
}

void KaldiRecognizer::AcceptWaveform(const float *fdata, int len)
{
    Vector<BaseFloat> wave;
    wave.Resize(len, kUndefined);
    for (int i = 0; i < len; i++)
        wave(i) = fdata[i];
    AcceptWaveform(wave);
}

void KaldiRecognizer::AcceptWaveform(Vector<BaseFloat> &wdata)
{
    lid_feature_->AcceptWaveform(sample_frequency_, wdata);
}

int KaldiRecognizer::Calculate() {
    frame_offset_ = 0;

    int num_frames = lid_feature_->NumFramesReady() - frame_offset_ * 3;
    Matrix <BaseFloat> features(num_frames, lid_feature_->Dim());

    for (int i = 0; i < num_frames; ++i) {
        Vector <BaseFloat> feat(lid_feature_->Dim());
        lid_feature_->GetFrame(i + frame_offset_ * 3, &feat);
        features.CopyRowFromVec(feat, i);
    }

    int32 compression_method_in = 1;
    CompressionMethod compression_method = static_cast<CompressionMethod>(
            compression_method_in);
    Matrix <BaseFloat> compressedMatrix;
    const CompressedMatrix &matrix = CompressedMatrix(features, compression_method);
    compressedMatrix.Resize(matrix.NumRows(), matrix.NumCols());
    matrix.CopyToMat(&compressedMatrix, kNoTrans);

    Matrix <BaseFloat> cmvn_feat(compressedMatrix.NumRows(),
                                 compressedMatrix.NumCols(), kUndefined);

    SlidingWindowCmn(lid_model_->sliding_opts, compressedMatrix, &cmvn_feat);

    int32 num_done = 0, num_err = 0;
    int32 num_unvoiced = 0;
    double tot_length = 0.0, tot_decision = 0.0;
    bool omit_unvoiced_utts = false;

    Vector <BaseFloat> vad_result(compressedMatrix.NumRows());

    ComputeVadEnergy(lid_model_->opts, compressedMatrix, &vad_result);

    double sum = vad_result.Sum();
    if (sum == 0.0) {
        KALDI_WARN << "No frames were judged voiced for utterance default";
        num_unvoiced++;
    } else {
        num_done++;
    }
    tot_decision += vad_result.Sum();
    tot_length += vad_result.Dim();

    const Vector <BaseFloat> &voiced = vad_result;

    if (cmvn_feat.NumRows() != voiced.Dim()) {
        KALDI_WARN << "Mismatch in number for frames " << cmvn_feat.NumRows()
                   << " for features and VAD " << voiced.Dim()
                   << ", for utterance default ";
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

    if (dim < MIN_LANG_FEATS) {
        return 1;
    }

    Matrix <BaseFloat> voiced_feat(dim, cmvn_feat.NumCols());
    int32 index = 0;
    for (int32 i = 0; i < cmvn_feat.NumRows(); i++) {
        if (voiced(i) != 0.0) {
            KALDI_ASSERT(voiced(i) == 1.0); // should be zero or one.
            voiced_feat.Row(index).CopyFromVec(cmvn_feat.Row(i));
            index++;
        }
    }
    KALDI_ASSERT(index == dim);

    Nnet3XvectorCompute(voiced_feat);

    PldaScoring();

    return 0;
}

const char *KaldiRecognizer::LangResult() {

    int res = Calculate();

    if (res != 0) {
        lang_result_ = "[]";
        return lang_result_.c_str();
    }
    using pair_type = decltype(scores_)::value_type;
    auto pr = std::max_element
    (
        std::begin(scores_), std::end(scores_), [] (const pair_type & p1, const pair_type & p2) {
            return p1.second < p2.second;
        }
    );

    KALDI_LOG << "key " << GetLanguage(pr -> first) << " value " << pr -> second;

    json::JSON obj;
    for (auto const &x : scores_) {
        json::JSON res;
        res["language"] = x.first;
        res["score"] = x.second;
        obj.append(res);
    }

    scores_.clear();
    lang_result_ = obj.dump();
    return lang_result_.c_str();

}

std::string KaldiRecognizer::GetLanguage(std::string lg) {
    std::map<std::string,std::string> languages = {
                {"ab","Abkhazian"},
                {"af","Afrikaans"},
                {"am","Amharic"},
                {"ar","Arabic"},
                {"as","Assamese"},
                {"az","Azerbaijani"},
                {"ba","Bashkir"},
                {"be","Belarusian"},
                {"bg","Bulgarian"},
                {"bn","Bengali"},
                {"bo","Tibetan"},
                {"br","Breton"},
                {"bs","Bosnian"},
                {"ca","Catalan"},
                {"ceb","Cebuano"},
                {"cs","Czech"},
                {"cy","Welsh"},
                {"da","Danish"},
                {"de","German"},
                {"el","Greek"},
                {"en","English"},
                {"eo","Esperanto"},
                {"es","Spanish"},
                {"et","Estonian"},
                {"eu","Basque"},
                {"fa","Persian"},
                {"fi","Finnish"},
                {"fo","Faroese"},
                {"fr","French"},
                {"gl","Galician"},
                {"gn","Guarani"},
                {"gu","Gujarati"},
                {"gv","Manx"},
                {"ha","Hausa"},
                {"haw","Hawaiian"},
                {"hi","Hindi"},
                {"hr","Croatian"},
                {"ht","Haitian"},
                {"hu","Hungarian"},
                {"hy","Armenian"},
                {"ia","Interlingua"},
                {"id","Indonesian"},
                {"is","Icelandic"},
                {"it","Italian"},
                {"iw","Hebrew"},
                {"ja","Japanese"},
                {"jw","Javanese"},
                {"ka","Georgian"},
                {"kk","Kazakh"},
                {"km","Central Khmer"},
                {"kn","Kannada"},
                {"ko","Korean"},
                {"la","Latin"},
                {"lb","Luxembourgish"},
                {"ln","Lingala"},
                {"lo","Lao"},
                {"lt","Lithuanian"},
                {"lv","Latvian"},
                {"mg","Malagasy"},
                {"mi","Maori"},
                {"mk","Macedonian"},
                {"ml","Malayalam"},
                {"mn","Mongolian"},
                {"mr","Marathi"},
                {"ms","Malay"},
                {"mt","Maltese"},
                {"my","Burmese"},
                {"ne","Nepali"},
                {"nl","Dutch"},
                {"nn","Norwegian Nynorsk"},
                {"no","Norwegian"},
                {"oc","Occitan"},
                {"pa","Panjabi"},
                {"pl","Polish"},
                {"ps","Pushto"},
                {"pt","Portuguese"},
                {"ro","Romanian"},
                {"ru","Russian"},
                {"sa","Sanskrit"},
                {"sco","Scots"},
                {"sd","Sindhi"},
                {"si","Sinhala"},
                {"sk","Slovak"},
                {"sl","Slovenian"},
                {"sn","Shona"},
                {"so","Somali"},
                {"sq","Albanian"},
                {"sr","Serbian"},
                {"su","Sundanese"},
                {"sv","Swedish"},
                {"sw","Swahili"},
                {"ta","Tamil"},
                {"te","Telugu"},
                {"tg","Tajik"},
                {"th","Thai"},
                {"tk","Turkmen"},
                {"tl","Tagalog"},
                {"tr","Turkish"},
                {"tt","Tatar"},
                {"uk","Ukrainian"},
                {"ur","Urdu"},
                {"uz","Uzbek"},
                {"vi","Vietnamese"},
                {"war","Waray"},
                {"yi","Yiddish"},
                {"yo","Yoruba"},
                {"zh","Chinese"}
    };

    return languages[lg];
}