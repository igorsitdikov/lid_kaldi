package l2m.recognition.language;

import com.sun.jna.PointerType;

public class Model extends PointerType implements AutoCloseable {
    public Model() {
    }

    public Model(String path) {
        super(LibLid.l2m_lid_model_new(path));
    }

    @Override
    public void close() {
        LibLid.l2m_lid_model_free(this.getPointer());
    }
}

