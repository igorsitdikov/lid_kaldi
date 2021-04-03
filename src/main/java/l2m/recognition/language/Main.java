package l2m.recognition.language;

import com.fasterxml.jackson.databind.ObjectMapper;
import lombok.SneakyThrows;

import java.io.File;
import java.nio.file.Files;
import java.util.Arrays;
import java.util.Comparator;
import java.util.List;
import java.util.stream.Collectors;

public class Main {

    private static final Integer TOP = 3;

    @SneakyThrows
    public static void main(String[] args) {
        final File file = new File("test_ru.wav");
        final Model lidModel = new Model("lid-model");
        final byte[] array = Files.readAllBytes(file.toPath());

        final byte[] data = new byte[array.length - 44];
        if (array.length - 44 >= 0) {
            System.arraycopy(array, 44, data, 0, array.length - 44);
        }
        final Recognizer recognizer = new Recognizer(lidModel, 8000);
        recognizer.acceptWaveForm(data);
        final String result = recognizer.getResult()
            .replaceAll("(\\d+),(\\d+)", "$1.$2");

        final List<LangScore> langScoreList = Arrays
            .stream(new ObjectMapper().readValue(result, LangScore[].class))
            .sorted(Comparator.comparing(LangScore::getScore).reversed())
            .limit(TOP)
            .collect(Collectors.toList());
        System.out.println(langScoreList);
    }
}
