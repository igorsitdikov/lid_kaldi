import os
import sys

from .lid_cffi import ffi as _ffi

def open_dll():
    dlldir = os.path.abspath(os.path.dirname(__file__))
    if sys.platform == 'win32':
        # We want to load dependencies too
        os.environ["PATH"] = dlldir + os.pathsep + os.environ['PATH']
        if hasattr(os, 'add_dll_directory'):
            os.add_dll_directory(dlldir)
        return _ffi.dlopen(os.path.join(dlldir, "liblid.dll"))
    elif sys.platform == 'linux':
        return _ffi.dlopen(os.path.join(dlldir, "liblid.so"))
    elif sys.platform == 'darwin':
        return _ffi.dlopen(os.path.join(dlldir, "liblid.dyld"))
    else:
        raise TypeError("Unsupported platform")

_c = open_dll()

class Model(object):

    def __init__(self, model_path):
        self._handle = _c.l2m_lid_model_new(model_path.encode('utf-8'))

    def __del__(self):
        _c.l2m_lid_model_free(self._handle)

class KaldiRecognizer(object):

    def __init__(self, *args):
        if len(args) == 2:
            self._handle = _c.l2m_recognizer_new_lid(args[0]._handle, args[1])
        else:
            raise TypeError("Unknown arguments")

    def __del__(self):
        _c.l2m_recognizer_free(self._handle)

    def AcceptWaveform(self, data):
        return _c.l2m_recognizer_accept_waveform(self._handle, data, len(data))

    def Result(self):
        return _ffi.string(_c.l2m_recognizer_lang_result(self._handle)).decode('utf-8')

