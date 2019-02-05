#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import sys
import base64
import hashlib
import configparser
from Crypto.Cipher import AES
from util import Utilty


class Cipher:
    def __init__(self, key, block_size=32):
        # Read config.ini.
        self.utility = utility
        config = configparser.ConfigParser()
        self.file_name = os.path.basename(__file__)
        self.full_path = os.path.dirname(os.path.abspath(__file__))
        config.read(os.path.join(self.full_path, 'config.ini'))

    def encrypt(self, raw_data, key, iv):
        # Encode raw data to base64.
        raw_data_base64 = base64.b64encode(raw_data)

        # Check length (can separate per 16byte).
        if len(raw_data_base64) % 16 != 0:
            raw_data_base64_16byte = raw_data_base64
            for i in range(16 - (len(raw_data_base64) % 16)):
                # Padding.
                raw_data_base64_16byte += b'_'
        else:
            raw_data_base64_16byte = raw_data_base64

        # Hashing secret key and initial vector.
        secret_key = hashlib.sha256(key.encode('utf-8')).digest()
        iv = hashlib.md5(iv.encode('utf-8')).digest()

        # Encrypt.
        crypto = AES.new(secret_key, AES.MODE_CBC, iv)
        cipher_data = crypto.encrypt(raw_data_base64_16byte)
        cipher_data_base64 = base64.b64encode(cipher_data)
        return cipher_data_base64

    def decrypt(self, cipher_data_base64, key, iv):
        # Decode base64 encoded data.
        cipher_data = base64.b64decode(cipher_data_base64)

        # Hashing secret key and initial vector.
        secret_key = hashlib.sha256(key.encode('utf-8')).digest()
        iv = hashlib.md5(iv.encode('utf-8')).digest()

        # Decrypt.
        crypto = AES.new(secret_key, AES.MODE_CBC, iv)
        raw_data_base64_16byte = crypto.decrypt(cipher_data)
        raw_data_base64 = raw_data_base64_16byte.split(b'_')[0]
        raw_data = base64.b64decode(raw_data_base64)
        return raw_data


if __name__ == '__main__':
    utility = Utilty()

    # Create secret key and initial vector.
    key = utility.get_random_token(50)
    iv = utility.get_random_token(10)

    # Create cipher instance.
    cipher = Cipher(key)

    # Load target binary file.
    with open('test.jpg', 'rb') as fin:
        raw_content = fin.read()
        print(raw_content)

    # Encrypt.
    encrypted = cipher.encrypt(raw_content, key, iv)
    with open('encrypt.bin', 'wb') as fout:
        fout.write(encrypted)

    # Decrypt.
    with open('encrypt.bin', 'rb') as fin:
        load_file = fin.read()
    decrypted = cipher.decrypt(load_file, key, iv)

    # Save decrypt data.
    with open('decrypted_test.jpg', 'wb') as fout:
        fout.write(decrypted)
