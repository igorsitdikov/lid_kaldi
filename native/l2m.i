%module(package="lid") lid

%include <typemaps.i>
%include <std_string.i>

#if SWIGPYTHON
%include <pybuffer.i>
#elif SWIGJAVA
%include <various.i>
#elif SWIGCSHARP
%include <arrays_csharp.i>
#endif

namespace kaldi {
    namespace nnet3 {
    }
}

#if SWIGPYTHON
%pybuffer_binary(const char *data, int len);
%ignore KaldiRecognizer::AcceptWaveform(const short *sdata, int len);
%ignore KaldiRecognizer::AcceptWaveform(const float *fdata, int len);
%exception {
  try {
    $action
  } catch (kaldi::KaldiFatalError &e) {
    PyErr_SetString(PyExc_RuntimeError, const_cast<char*>(e.KaldiMessage()));
    SWIG_fail;
  } catch (std::exception &e) {
    PyErr_SetString(PyExc_RuntimeError, const_cast<char*>(e.what()));
    SWIG_fail;
  }
}
#endif
#if SWIGJAVA
%apply char *BYTE {const char *data};
%typemap(javaimports) KaldiRecognizer %{
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
%}
%typemap(javacode) KaldiRecognizer %{
  public void AcceptWaveform(byte[] data) {
    AcceptWaveform(data, data.length);
  }
  public void AcceptWaveform(short[] data, int len) {
    byte[] bdata = new byte[len * 2];
    ByteBuffer.wrap(bdata).order(ByteOrder.LITTLE_ENDIAN).asShortBuffer().put(data, 0, len);
    AcceptWaveform(bdata, bdata.length);
  }
%}
%pragma(java) jniclasscode=%{
    static {
        boolean loadedEmbeddedLibrary = EmbeddedLibraryTools.LOADED_EMBEDDED_LIBRARY;
    }
%}
#endif

%{
#include "lid_api.h"
typedef struct L2mLidModel LidModel;
typedef struct L2mRecognizer KaldiRecognizer;
%}

typedef struct {} LidModel;
typedef struct {} KaldiRecognizer;


%extend LidModel {
    LidModel(const char *model_path)  {
        return l2m_lid_model_new(model_path);
    }
    ~LidModel() {
        l2m_lid_model_free($self);
    }
}
%extend KaldiRecognizer {
    KaldiRecognizer(LidModel *model, float sample_rate)  {
        return l2m_recognizer_new_lid(model, sample_rate);
    }
    ~KaldiRecognizer() {
        l2m_recognizer_free($self);
    }
    void AcceptWaveform(const char *data, int len) {
        return l2m_recognizer_accept_waveform($self, data, len);
    }
    const char* LangResult() {
        return l2m_recognizer_lang_result($self);
    }
}

%rename(SetLogLevel) lid_set_log_level;
void lid_set_log_level(int level);