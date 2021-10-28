import os
import cv2 as cv
from pytesseract import image_to_string
from PIL import Image
from PIL import ImageFilter
from PIL import ImageOps
from PIL import ImageEnhance
import timeit
import numpy as np
import pandas as pd
from pathlib import Path

# Start Timer
start = timeit.default_timer()

# Jiaming's Data
# Path to the data directory
# data_dir = Path("./CAPTCHAS_DATASET/")
# data = pd.read_csv("./CAPTCHAS_DATASET/captcha_label.csv")
# # Get list of all the images
# images = sorted(list(map(str, list(data_dir.glob("*.png")))))
# labels = pd.DataFrame(data, columns=['Ground_Truth'])
# characters = "0123456789ABCDEFGHJKLMNOPRSTUVWXZabcdefghijklmnopqrstuvwxyz"

# 1024 Test data
data_dir = Path("./OCR_DEMO_DATASET/")
images = sorted(list(map(str, list(data_dir.glob("*.jpg")))))
labels = [img.split(os.path.sep)[-1].split(".jpg")[0] for img in images]
characters = set(char for label in labels for char in label)

print("Number of images found: ", len(images))
print("Number of labels found: ", len(labels))
print("Number of unique characters: ", len(characters))
print("Characters present: ", characters)

#This function returns the result of Tesseract
def ocr(im):
    return image_to_string(im, lang="Captcha", config="--psm 10 -c tessedit_char_whitelist=0123456789ABCDEFGHJKLMNOPRSTUVWXZabcdefghijklmnopqrstuvwxyz")

#Removing Background Color From Left to Right
def rem_back(trg):
    cnt = ImageEnhance.Contrast(trg)
    trg = trg.filter(ImageFilter.MedianFilter(3))
    trg = cnt.enhance(1)
    mat = trg.load()
    tol = 10
    cn = 0
    for r in range(0, trg.height):
        temp = mat[0, r][0]
        for c in range(0, trg.width):
            if (mat[c, r][0] <= temp + tol and mat[c, r][0] >= temp - tol):
                mat[c, r] = (255, 255)
    return trg

#Removing Background Color From Right to Left
def rem_back_rev(trg):
    cnt = ImageEnhance.Contrast(trg)
    trg = trg.filter(ImageFilter.MedianFilter(3))
    trg = cnt.enhance(1)
    mat = trg.load()
    tol = 10
    cn = 0
    for r in range(trg.height - 1, -1, -1):
        temp = mat[trg.width - 1, r][0]
        for c in range(trg.width - 1, -1, -1):
            if (mat[c, r][0] <= temp + tol and mat[c, r][0] >= temp - tol):
                mat[c, r] = (255, 255)
    return trg


#Making Characters ready for OCR
def char_op(trg):
    trg = rem_back(trg)
    trg = rem_back_rev(trg)
    cnt = ImageEnhance.Contrast(trg)
    trg = cnt.enhance(2)
    trg = trg.filter(ImageFilter.MedianFilter(9))
    trg = trg.filter(ImageFilter.RankFilter(5,3))
    br = ImageEnhance.Brightness(trg)
    trg = br.enhance(1)
    trg = trg.convert("RGB")
    trg = ImageOps.expand(trg, 30, 'white')
    return trg
    
with open('result_pure_ocr_demo.txt', 'w') as f:
    count = 0
    for i in images:
        # Pillow solution
        im = Image.open(i).convert("LA")
        im = im.resize((5 * im.width, 5 * im.height), Image.ANTIALIAS)
        im = im.filter(ImageFilter.MedianFilter(3))
        im = im.filter(ImageFilter.GaussianBlur(3))
        im = im.filter(ImageFilter.MedianFilter(5))
        im = im.filter(ImageFilter.SMOOTH_MORE)
        cnt = ImageEnhance.Contrast(im)
        im = cnt.enhance(1)

        width, height = im.size
        sub_width = width / 9
        a = im.crop((0, 0, 3*sub_width, height))
        b = im.crop((3*sub_width, 0, 4*sub_width, height))
        c = im.crop((4*sub_width, 0, 5*sub_width, height))
        d = im.crop((5*sub_width, 0, 6*sub_width, height))
        e = im.crop((6*sub_width, 0, width, height))

        a = char_op(a)
        b = char_op(b)
        c = char_op(c)
        d = char_op(d)
        e = char_op(e)
        
        pred = "".join(ocr(a).rstrip() + ocr(b).rstrip() + ocr(c).rstrip() +
            ocr(d).rstrip() + ocr(e).rstrip())

        print(pred, (i.split('/')[-1]).split('.')[0])
        f.write("pred: %s, true: %s\n" % (pred, (i.split('/')[-1]).split('.')[0]))
    

stop = timeit.default_timer()
print('Time: ', stop - start)  