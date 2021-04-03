package l2m.recognition.language;

import com.sun.jna.PointerType;

public class Recognizer extends PointerType implements AutoCloseable {
    public Recognizer(Model model, float sampleRate) {
        super(LibLid.l2m_recognizer_new_lid(model, sampleRate));
    }


    public void acceptWaveForm(byte[] data) {
        LibLid.l2m_recognizer_accept_waveform(this.getPointer(), data, data.length);
    }

    public void acceptWaveForm(short[] data) {
        LibLid.l2m_recognizer_accept_waveform_s(this.getPointer(), data, data.length);
    }

    public void acceptWaveForm(float[] data) {
        LibLid.l2m_recognizer_accept_waveform_f(this.getPointer(), data, data.length);
    }

    public String getResult() {
        return LibLid.l2m_recognizer_lang_result(this.getPointer());
    }

    @Override
    public void close() {
        LibLid.l2m_recognizer_free(this.getPointer());
    }
}
