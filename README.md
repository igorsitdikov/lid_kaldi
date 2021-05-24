# Spoken Language Recognition based on Kaldi


## Linux installation
1. Use `pip` to install corresponding to your platform `whl` package from https://github.com/igorsitdikov/lid_kaldi/releases

Example:
```
pip3 install https://github.com/igorsitdikov/lid_kaldi/releases/download/latest/lid-1.0.1-py3-none-linux_x86_64.whl
```

2. Download `lid-model.tar.xz` model from https://github.com/igorsitdikov/lid_kaldi/releases and unpack to your project folder
```
wget https://github.com/igorsitdikov/lid_kaldi/releases/download/1.0.1/lid-model.tar.xz
tar -xvf lid-model.tar.xz
```
3. Run example script (not required)
```
wget https://github.com/igorsitdikov/lid_kaldi/raw/master/python/example/test_simple.py
wget https://github.com/igorsitdikov/lid_kaldi/raw/master/python/example/test_ru.wav
python3 test_simple.py test_ru.wav
```
