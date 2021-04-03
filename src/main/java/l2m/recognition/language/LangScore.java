package l2m.recognition.language;

import lombok.Data;

@Data
public class LangScore {

    private String language;
    private Double score;

    @Override
    public String toString() {
        return String.format("{'language':'%s', 'score':%s}", LanguageMapper.getLanguage(language), score);
    }
}
