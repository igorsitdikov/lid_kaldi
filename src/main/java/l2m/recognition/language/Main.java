package l2m.recognition.language;

import com.fasterxml.jackson.databind.ObjectMapper;
import lombok.SneakyThrows;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.util.*;
import java.util.stream.Collectors;

public class Main {

    @SneakyThrows
    public static void main(String[] args) {
        System.out.println("current platform:       " + EmbeddedLibraryTools.getCurrentPlatformIdentifier());
        String pathToFile;
        String model;
        if (args.length != 2) {
            System.out.println("java -jar file.jar <wav> <model>");
            pathToFile = "test_ru.wav";
            model = "lid-107";
        } else {
            for (int i = 0; i < args.length; i++) {
                System.out.println(i + " " +args[i]);
            }
            pathToFile = args[0];
            model = args[1];
        }

        lid.SetLogLevel(-10);
        final File file = new File(pathToFile);
        LidModel lidModel = new LidModel(model);
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
        recognizer.AcceptWaveform(data);
        String result = recognizer.LangResult().replaceAll("(\\d+),(\\d+)", "$1.$2");

        List<LangScore> ppl2 = Arrays.asList(new ObjectMapper().readValue(result, LangScore[].class));
        Collections.sort(ppl2, Comparator.comparing(LangScore::getScore));
        if (ppl2.size() != 0) {
            List<LangScore> top3 = new ArrayList<>(ppl2.subList(ppl2.size() - 5, ppl2.size()));
            System.out.println(top3
                    .stream()
                    .peek(el -> el.setLanguage(LanguageMap.getLanguage(el.getLanguage())))
                    .collect(Collectors.toList()));
        }
    }
}