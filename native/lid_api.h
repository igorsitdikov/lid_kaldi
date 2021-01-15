#ifndef _L2M_API_H_
#define _L2M_API_H_

#ifdef __cplusplus
extern "C" {
#endif

typedef struct L2mLidModel L2mLidModel;
typedef struct L2mRecognizer L2mRecognizer;

L2mLidModel *l2m_lid_model_new(const char *model_path);
void l2m_lid_model_free(L2mLidModel *model);

L2mRecognizer *l2m_recognizer_new_lid(L2mLidModel *lid_model, float sample_rate);
void l2m_recognizer_accept_waveform(L2mRecognizer *recognizer, const char *data, int length);
void vosk_recognizer_accept_waveform_s(L2mRecognizer *recognizer, const short *data, int length);
void vosk_recognizer_accept_waveform_f(L2mRecognizer *recognizer, const float *data, int length);
const char *l2m_recognizer_lang_result(L2mRecognizer *recognizer);
void l2m_recognizer_free(L2mRecognizer *recognizer);
void lid_set_log_level(int log_level);
#ifdef __cplusplus
}
#endif

#endif /* _L2M_API_H_ */
