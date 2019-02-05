# Broken Face Recognition
**The system is broken face recognition for adversarial example testing.**

## Overview
TBA

 <img src="./img/auth_sample.jpg" width="500">

## Installation
 1. git clone Broken FC's repository.  
 ```
 root@kali:~# git clone https://github.com/13o-bbr-bbq/Broken_FaceRecognition.git
 ```

 2. Get Python3-pip.  
 ```
 root@kali:~# apt-get update
 root@kali:~# apt-get install python3-pip
 ```

 3. Install required python's package.  
 ```
 root@kali:~# cd Broken_FaceRecognition
 root@kali:~/Broken_FaceRecognition# pip3 install -r requirements.txt
 ```

## Usage
```
./broken_face_recognition.py
usage:
    ./broken_face_recognition.py [-t] [-g <label_name>] [-c]
    ./broken_face_recognition.py -h | --help
options:
    -t   Optional : Train face recognition model.
    -g   Optional : Gather your face images.
    -c   Optional : Create dataset that train and test.
    -h --help     Show this help message and exit.
```



### 1. Face authorization (No option).
This mode is default that without option.  

```
root@kali:~/Broken_FaceRecognition# python3 broken_face_recognition.py
```

### 2. Train face recognition model.

```
root@kali:~/Broken_FaceRecognition# python3 broken_face_recognition.py -t
```

### 3. Gather your face images. 
```
root@kali:~/Broken_FaceRecognition# python3 broken_face_recognition.py -g "Any label name"
```

### 4. Create dataset.
```
root@kali:~/Broken_FaceRecognition# python3 broken_face_recognition.py -c
```

## Operation check environment
 * Hardware  
   * OS: Kali Linux 2018.2  
   * CPU: Intel(R) Core(TM) i7-6500U 2.50GHz  
   * GPU: None  
   * Memory: 8.0GB  
 * Software  
   * Metasploit Framework 4.16.48-dev
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
