# American Sign Language Reading Using Mediapipe!

## IMPORTANT NOTE: mediapipe does not work with some new python versions for some stupid reason, so you should use Python3.11 if possible:
[https://www.python.org/downloads/release/python-3110/](https://www.python.org/downloads/release/python-3110/ "")
```powershell
& "$env:USERPROFILE\AppData\Local\Programs\Python\Python311\python.exe" .\AslRecognizeGui.py
```

The project already has trained models in the `/models/` folder, which you can run using `main.py`.


If you want to train from scratch:

Create ".csv" files for the label you are training for using `collect_data.py`
```
python collect_data.py
```
Train a model using these csvs using `train_classifier`:
```bash
python train_classifier.py
```
Test using `AslRecognizeGui.py` which expects models to be named "best":
```bash
python AslRecognizeGui.py
```