//
// Created by sitdikov on 23.09.20.
//

#include "src/vosk_api.h"
#include <stdio.h>

int main() {



    char ch_arr[3][29] = {
            "common_voice_ru_22498222.wav",
            "common_voice_ru_22498222.wav",
            "common_voice_ru_22498222.wav"
    };

    VoskLidModel *lid_model = vosk_lid_model_new("model-lid");

    for(int i = 0; i < 3; i++) {
               VoskRecognizer *recognizer = vosk_recognizer_new_lid(lid_model, 8000.0);
        FILE *wavin;
        int nread, final;
        if (fopen((const  char *)(ch_arr + i), "rb") != NULL) {
            wavin = fopen((const  char *)(ch_arr + i), "rb");
            fseek(wavin, 0L, SEEK_END);
            long sz = ftell(wavin);
            fseek(wavin, 0L, SEEK_SET);
            char buf[sz - 44];
            fseek(wavin, 44, SEEK_SET);

            nread = fread(buf, 1, sizeof(buf), wavin);
            printf("%s\n", vosk_recognizer_lang_result(recognizer, buf, nread));

            vosk_recognizer_free(recognizer);

            fclose(wavin);
        }
    }
    vosk_lid_model_free(lid_model);
    return 0;
}