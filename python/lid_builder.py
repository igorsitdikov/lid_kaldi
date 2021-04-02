#!/usr/bin/env python3

import os
from cffi import FFI

lid_root=os.environ.get("LID_SOURCE", "..")
cpp_command = "cpp " + lid_root + "/native/lid_api.h"

ffibuilder = FFI()
ffibuilder.set_source("lid.lid_cffi", None)
ffibuilder.cdef(os.popen(cpp_command).read())

if __name__ == '__main__':
    ffibuilder.compile(verbose=True)

