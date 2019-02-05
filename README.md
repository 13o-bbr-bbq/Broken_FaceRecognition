# Broken Face Recognition
**The system is broken face recognition for adversarial example testing.**

## Overview
TBA

 <img src="./img/auth_sample.jpg" width="500">

## Installation
 1. git clone Broken FR's repository.  
 ```
 babaroa@ubuntu:~$ git clone https://github.com/13o-bbr-bbq/Broken_FaceRecognition.git
 ```

 2. Get Python3-pip.  
 ```
 babaroa@ubuntu:~$ apt-get update
 babaroa@ubuntu:~$ apt-get install python3-pip
 ```

 3. Install required python's package.  
 ```
 babaroa@ubuntu:~$ cd Broken_FaceRecognition
 babaroa@ubuntu:~/Broken_FaceRecognition$ pip3 install -r requirements.txt
 ```

 4. Download default trained model.  
 If you want to use the default model, please download it.  
 ```
 babaroa@ubuntu:~/Broken_FaceRecognition$ wget "https://drive.google.com/uc?export=download&id=14UUBAkpf3FJ0KW_awotEkGZelqrH13M1" -O model.zip
 babaroa@ubuntu:~/Broken_FaceRecognition$ unzip model.zip
 babaroa@ubuntu:~/Broken_FaceRecognition$ mv finetuning.h5 dataset/model/
 babaroa@ubuntu:~/Broken_FaceRecognition$ rm model.zip
```

## Usage
```
./broken_face_recognition.py
usage:
    ./broken_face_recognition.py [-g <label_name>] [-c] [-t]
    ./broken_face_recognition.py -h | --help
options:
    -g   Optional : Gather your face images.
    -c   Optional : Create dataset that train and test.
    -t   Optional : Train face recognition model.
    -h --help     Show this help message and exit.
```

If you don't indicate option, this system is operated "face recognition" mode (Please refer #1).  

### 1. Face recognition (No option).
```
babaroa@ubuntu:~/Broken_FaceRecognition$ python3 broken_face_recognition.py
```

<img src="./img/auth_sample.jpg" width="200">

When the recognition rate exceeds the threshold value, the string "Unlock" is displayed on the window.  

|Note|
|:---|
|If you want to change the threshold value, please edit `threshold` value in the `config.ini`.|

### 2. Gather your face images. 
```
babaroa@ubuntu:~/Broken_FaceRecognition$ python3 broken_face_recognition.py -g "Any label name"
```

This system captures your faces using equiped camera of your PC per `cap_wait_time` (ms).  
The captured face images is stored in the "Any label name" directory under the `original_image`.  

```
babaroa@ubuntu:~/Broken_FaceRecogn/original_image/"Any label name"$ ls
captured_face_1.jpg
captured_face_2.jpg
captured_face_3.jpg
... snip ...
```

|Note|
|:---|
|If you want to change the `cap_wait_time` and sampling numbers of face, please edit `cap_wait_time` value and `gather_samples` value in the `config.ini`.|

### 3. Create dataset.
```
babaroa@ubuntu:~/Broken_FaceRecognition$ python3 broken_face_recognition.py -c
```

This system create train data and test data from numerous face images under the `original_image` directory.  
And, the created data are putted on the `train` and `test` directory under the `dataset`.

```
babaroa@ubuntu:~/Broken_FaceRecogn/dataset$ ls
train
test
```

### 4. Train face recognition model.
```
babaroa@ubuntu:~/Broken_FaceRecognition$ python3 broken_face_recognition.py -t
```

This system learns the features of numerous face images using trainning data.  
And, the learned result is stored on the `model` directory under the `dataset`.  

```
babaroa@ubuntu:~/Broken_FaceRecogn/dataset/model$ ls
finetuning.h5
```

|Note|
|:---|
|If you want to change the model name, please edit `model_name` in the `config.ini`.|

## Operation check environment
 * Hardware  
   * OS: Windows 10 64bit  
   * CPU: Intel(R) Core(TM) i7-6500U 2.50GHz  
   * GPU: NVIDIA GeForce m975  
   * Memory: 16.0GB  
 * Software  
   * Python 3.6.1
   * docopt==0.6.2
   * Keras==2.2.4
   * numpy==1.16.1
   * opencv-python==4.0.0.21
   * tensorflow==1.8.0

## License
[Apache License 2.0](https://github.com/13o-bbr-bbq/Broken_FaceRecognition/blob/master/LICENSE)

## Contact us
Isao Takaesu  
takaesu235@gmail.com  
[https://twitter.com/bbr_bbq](https://twitter.com/bbr_bbq)
