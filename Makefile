KALDI_ROOT=/opt/kaldi
CFLAGS := -g -O2 -DPIC -fPIC -Wno-unused-function
JAVA_HOME=/usr/lib/jvm/java-1.8.0-openjdk.x86_64
CXX := g++

ifeq ($(OS),Windows_NT)
	TARGET := liblid.dll
	JAVA_OS := win32
	ifeq ($(PROCESSOR_ARCHITECTURE),AMD64)
		OS_PATH := amd64/Windows
	endif
else
    TARGET := liblid.so
	JAVA_OS := linux
	UNAME_S := $(shell uname -s)
	UNAME_P := $(shell uname -p)
	ifeq ($(UNAME_S),Linux)
		ifeq ($(UNAME_P),x86_64)
		    OS_PATH := amd64/Linux
    	endif
	endif
endif

CPPFLAGS := -I$(JAVA_HOME)/include -I$(JAVA_HOME)/include/$(JAVA_OS) -I$(KALDI_ROOT)/src -I$(KALDI_ROOT)/tools/openfst/include -I./native
OUTPUT_PATH := src/main/resources/NATIVE/$(OS_PATH)/

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

all: $(TARGET) copy

VOSK_SOURCES = \
	lid_wrap.cc \
	native/kaldi_recognizer.cc \
	native/kaldi_recognizer.h \
	native/lid_model.cc \
	native/lid_model.h \
	native/lid_api.cc \
	native/lid_api.h

copy:
	strip $(TARGET)
	mkdir -p $(OUTPUT_PATH)
	cp $(TARGET) $(OUTPUT_PATH)
	chmod +x copy_dependencies.sh
	./copy_dependencies.sh $(TARGET) $(OUTPUT_PATH)

$(TARGET): $(VOSK_SOURCES)
	$(CXX) -shared -o $@ $(CPPFLAGS) $(CFLAGS) $(VOSK_SOURCES) $(KALDI_LIBS)

lid_wrap.cc: native/l2m.i
	mkdir -p src/main/java/l2m/recognition/language
	swig -c++ -I./native \
		-java -package l2m.recognition.language \
		-outdir src/main/java/l2m/recognition/language -o $@ $<

mvn:
	mvn clean package
	mvn install:install-file -Dfile="target/lid-jni-1.0.1-jar-with-dependencies.jar" -DgroupId=l2m.asr -DartifactId=lid-jni -Dversion=1.0.1 -Dpackaging=jar -DgeneratePom=true

clean:
	$(RM) *.so *_wrap.cc *.o *.a native/*.o *_wrap.o liblid.$(EXTENSION)
	$(RM) -r target