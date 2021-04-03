package l2m.recognition.language;

import com.sun.jna.Native;
import com.sun.jna.Platform;
import com.sun.jna.Pointer;

public class LibLid {

    static {

        Native.register(LibLid.class, Platform.isWindows() ? "liblid" : "lid");
    }

    public static native Pointer l2m_lid_model_new(String path);

    public static native void l2m_lid_model_free(Pointer model);

    public static native Pointer l2m_recognizer_new_lid(Model model, float sample_rate);

    public static native void l2m_recognizer_accept_waveform(Pointer recognizer, byte[] data, int length);

    public static native void l2m_recognizer_accept_waveform_s(Pointer recognizer, short[] data, int length);

    public static native void l2m_recognizer_accept_waveform_f(Pointer recognizer, float[] data, int length);

    public static native String l2m_recognizer_lang_result(Pointer recognizer);

    public static native void l2m_recognizer_free(Pointer recognizer);

    public static native void lid_set_log_level(int log_level);
}
