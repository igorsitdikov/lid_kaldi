package l2m;

public class Trial {
    private String lang;
    private String path;

    public Trial(String lang, String path) {
        this.lang = lang;
        this.path = path;
    }

    public String getLang() {
        return lang;
    }

    public void setLang(final String lang) {
        this.lang = lang;
    }

    public String getPath() {
        return path;
    }

    public void setPath(final String path) {
        this.path = path;
    }
}
