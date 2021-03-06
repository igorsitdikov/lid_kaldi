import os
import sys
import setuptools
import shutil
import glob
import platform

# Figure out environment for cross-compile
lid_source = os.getenv("LID_SOURCE", os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
system = os.environ.get('LID_PLATFORM', platform.system())
architecture = os.environ.get('LID_ARCHITECTURE', platform.architecture()[0])

# Copy precompmilled libraries
for lib in glob.glob(os.path.join(lid_source, "native/lib*.*")):
    print ("Adding library", lib)
    shutil.copy(lib, "lid")

# Create OS-dependent, but Python-independent wheels.
try:
    from wheel.bdist_wheel import bdist_wheel
except ImportError:
    cmdclass = {}
else:
    class bdist_wheel_tag_name(bdist_wheel):
        def get_tag(self):
            abi = 'none'
            if system == 'Darwin':
                oses = 'macosx_10_6_x86_64'
            elif system == 'Windows' and architecture == '32bit':
                oses = 'win32'
            elif system == 'Windows' and architecture == '64bit':
                oses = 'win_amd64'
            elif system == 'Linux' and architecture == '64bit':
                oses = 'linux_x86_64'
            elif system == 'Linux':
                oses = 'linux_' + architecture
            else:
                raise TypeError("Unknown build environment")
            return 'py3', abi, oses
    cmdclass = {'bdist_wheel': bdist_wheel_tag_name}

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="lid",
    version="1.0.1",
    author="Igor Sitdikov",
    author_email="ihar.sitdzikau@yandex.ru",
    description="Spoken Language Identification",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/igorsitdikov/lid_kaldi",
    packages=setuptools.find_packages(),
    package_data = {'lid': ['*.so', '*.dll', '*.dyld']},
    include_package_data=True,
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: Microsoft :: Windows',
        'Operating System :: POSIX :: Linux',
        'Operating System :: MacOS :: MacOS X',
        'Topic :: Software Development :: Libraries :: Python Modules'
    ],
    cmdclass=cmdclass,
    python_requires='>=3',
    zip_safe=False, # Since we load so file from the filesystem, we can not run from zip file
    setup_requires=['cffi>=1.0'],
    install_requires=['cffi>=1.0'],
    cffi_modules=['lid_builder.py:ffibuilder'],
)
