package l2m;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;

public class Main {

    public static void main(String[] args) {
        Model model = new Model("model");
        LidModel lidModel = new LidModel("model-lid");
        SpkModel spkModel = new SpkModel("model-spk");
        final File file = new File("common_voice_ru_22498222.wav");
        byte[] ar = new byte[0];
        try {
            ar = Files.readAllBytes(file.toPath());
        } catch (IOException e) {
            e.printStackTrace();
        }
        byte[] data = new byte[ar.length - 44];
        for (int i = 44; i < ar.length; i++) {
            data[i-44] = ar[i];
        }
        KaldiRecognizer recognizer = new KaldiRecognizer(lidModel, 8000);
        System.out.println(recognizer.LangResult(data, data.length));
        KaldiRecognizer recognizer1 = new KaldiRecognizer(model, spkModel, 8000);
        recognizer1.AcceptWaveform(data, data.length);
        recognizer1.FinalResult();
    }

}


//[{
//    "language" : "eu",
//    "score" : -7,613651
//    }, {
//    "language" : "it",
//    "score" : -8,279074
//    }, {
//    "language" : "pl",
//    "score" : -7,616603
//    }, {
//    "language" : "ru",
//    "score" : 2,516520
//}]