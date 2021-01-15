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

#include "lid_api.h"
#include "kaldi_recognizer.h"
#include "lid_model.h"

#include <string.h>

using namespace kaldi;

L2mLidModel *l2m_lid_model_new(const char *model_path)
{
    return (L2mLidModel *)new LidModel(model_path);
}

void l2m_lid_model_free(L2mLidModel *model)
{
    ((LidModel *)model)->Unref();
}

L2mRecognizer *l2m_recognizer_new_lid(L2mLidModel *lid_model, float sample_rate)
{
    return (L2mRecognizer *)new KaldiRecognizer((LidModel *)lid_model, sample_rate);
}

void l2m_recognizer_accept_waveform(L2mRecognizer *recognizer, const char *data, int length)
{
    ((KaldiRecognizer *)(recognizer))->AcceptWaveform(data, length);
}

void l2m_recognizer_accept_waveform_s(L2mRecognizer *recognizer, const short *data, int length)
{
    ((KaldiRecognizer *)(recognizer))->AcceptWaveform(data, length);
}

void l2m_recognizer_accept_waveform_f(L2mRecognizer *recognizer, const float *data, int length)
{
    ((KaldiRecognizer *)(recognizer))->AcceptWaveform(data, length);
}

const char *l2m_recognizer_lang_result(L2mRecognizer *recognizer)
{
    return ((KaldiRecognizer *)recognizer)->LangResult();
}

void l2m_recognizer_free(L2mRecognizer *recognizer)
{
    delete (KaldiRecognizer *)(recognizer);
}

void lid_set_log_level(int log_level)
{
    SetVerboseLevel(log_level);
}