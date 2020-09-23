package l2m;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.ObjectMapper;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;
import java.util.stream.Collectors;

public class LidMain {
    private final static String PATH_TO_WAV_SCP = "wav.scp";
    private final static String PATH_TO_CORPUS = "/home/sitdikov/IdeaProjects/kaldi-asr/egs/lid_x/v2";
    public static void main(String[] args) throws IOException, InterruptedException {

        int correct = 0;
        int wrong = 0;
        final List<String> lines = Files.readAllLines(Path.of(PATH_TO_WAV_SCP));
//        Collections.reverse(lines);
        final List<Trial> trials = lines.stream().map(el -> {
            final String[] buf = el.split(" ");
            final String lang = buf[0].substring(0, 2);
            final String path = String.format("%s%s", PATH_TO_CORPUS, buf[1].substring(1));
            return new Trial(lang, path);
        }).collect(Collectors.toList());

        System.out.println();
        for (Trial el : trials) {
            l2m.LidModel lidModel = new l2m.LidModel("model-lid");
            final File file = new File(el.getPath());
            byte[] ar = new byte[0];
            try {
                ar = Files.readAllBytes(file.toPath());
            } catch (IOException e) {
                e.printStackTrace();
            }
            byte[] data = new byte[ar.length - 44];
            for (int i = 44; i < ar.length; i++) {
                data[i - 44] = ar[i];
            }
            l2m.KaldiRecognizer recognizer = new l2m.KaldiRecognizer(lidModel, 8000);
            recognizer.Calculate(ar, ar.length);
            String langResult = recognizer.LangResult(ar, ar.length).replaceAll("(\\d+),(\\d+)", "$1.$2");
            ObjectMapper mapper = new ObjectMapper();
            try {
                l2m.LangScore[] scores = mapper.readValue(langResult, l2m.LangScore[].class);
                List<l2m.LangScore> ppl2 = Arrays.asList(mapper.readValue(langResult, l2m.LangScore[].class));
                l2m.LangScore langScore =  Collections.max(ppl2, Comparator.comparing(s -> s.getScore()));
                if (langScore.getLanguage().equals(el.getLang())) {
                    correct++;
                    System.out.println(langScore.getLanguage() + " " + el.getLang());
                    printResults(correct, wrong);
                } else {
                    wrong++;
                    System.out.println(langScore.getLanguage() + " " + el.getLang());
                    printResults(correct, wrong);
                }
                System.out.println();
            } catch (JsonProcessingException e) {
                e.printStackTrace();
            }
//            recognizer.delete();
//            recognizer.finalize();
//            lidModel.Unref();
//            lidModel.delete();
//            lidModel = null;
//            recognizer = null;
//            if (correct % 50 == 0) {
//                System.gc();
//            }
            System.out.println(langResult);
        }
        System.out.println("***** total ******");
        printResults(correct, wrong);
    }


    private static void printResults(int correct, int wrong) {
        System.out.println("Correct: " + correct);
        System.out.println("Wrong: " + wrong);
        System.out.println("EER: " + (1.0 - (correct * 1.0 / (correct + wrong))));
    }
}
