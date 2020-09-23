KALDI_ROOT=/opt/kaldi
CFLAGS := -g -O2 -DPIC -fPIC -Wno-unused-function
JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64
CPPFLAGS := -I$(JAVA_HOME)/include -I$(JAVA_HOME)/include/linux -I$(KALDI_ROOT)/src -I$(KALDI_ROOT)/tools/openfst/include -I../src
#MKLROOT= /opt/intel/compilers_and_libraries/linux/mkl
#ATLASLIBS = /usr/lib/x86_64-linux-gnu/libatlas.so.3 /usr/lib/x86_64-linux-gnu/libf77blas.so.3 /usr/lib/x86_64-linux-gnu/libcblas.so.3 /usr/lib/x86_64-linux-gnu/liblapack_atlas.so.3 -Wl,-rpath=/usr/lib/x86_64-linux-gnu
ATLASLIBS := /usr/lib/libatlas.so.3 /usr/lib/libf77blas.so.3 /usr/lib/libcblas.so.3 /usr/lib/liblapack_atlas.so.3
CXX := g++

BUILD_DIR = $(PWD)
DEST_DIR = /usr/lib

COPY_FILES = $(DEST_DIR)/libvosk_jni_cpu.so

KALDI_LIBS = \
             ${KALDI_ROOT}/src/online2/kaldi-online2.a \
             ${KALDI_ROOT}/src/decoder/kaldi-decoder.a \
             ${KALDI_ROOT}/src/ivector/kaldi-ivector.a \
             ${KALDI_ROOT}/src/gmm/kaldi-gmm.a \
             ${KALDI_ROOT}/src/nnet3/kaldi-nnet3.a \
             ${KALDI_ROOT}/src/tree/kaldi-tree.a \
             ${KALDI_ROOT}/src/feat/kaldi-feat.a \
             ${KALDI_ROOT}/src/lat/kaldi-lat.a \
             ${KALDI_ROOT}/src/lm/kaldi-lm.a \
             ${KALDI_ROOT}/src/hmm/kaldi-hmm.a \
             ${KALDI_ROOT}/src/transform/kaldi-transform.a \
             ${KALDI_ROOT}/src/cudamatrix/kaldi-cudamatrix.a \
             ${KALDI_ROOT}/src/matrix/kaldi-matrix.a \
             ${KALDI_ROOT}/src/fstext/kaldi-fstext.a \
             ${KALDI_ROOT}/src/util/kaldi-util.a \
             ${KALDI_ROOT}/src/base/kaldi-base.a \
             ${KALDI_ROOT}/tools/openfst/lib/libfst.a \
             ${KALDI_ROOT}/tools/openfst/lib/libfstngram.a \
             ${KALDI_ROOT}/tools/OpenBLAS/libopenblas.a \
            -lm -lpthread \
            -lgfortran

all: libvosk_jni_cpu.so

VOSK_SOURCES = \
	vosk_wrap.cc \
	src/kaldi_recognizer.cc \
	src/kaldi_recognizer.h \
	src/model.cc \
	src/model.h \
	src/spk_model.cc \
	src/spk_model.h \
	src/lid_model.cc \
	src/lid_model.h \
	src/vosk_api.cc \
	src/vosk_api.h

copy: $(COPY_FILES)

$(DEST_DIR)/%.so: $(BUILD_DIR)/%.so
	sudo cp -f $< $@

libvosk_jni_cpu.so: $(VOSK_SOURCES)
	$(CXX) -shared -o $@ $(CPPFLAGS) $(CFLAGS) $(VOSK_SOURCES) $(KALDI_LIBS)

vosk_wrap.cc: src/vosk_.i
	mkdir -p src/main/java/l2m
	swig -c++ -I./src \
		-java -package l2m \
		-outdir src/main/java/l2m -o $@ $<

clean:
	$(RM) *.so *_wrap.cc *_wrap.o test/*.class
	$(RM) -r org model-en