import cv2 
import os
from superimposer import superimpose

import argparse

parser = argparse.ArgumentParser()

# Add arguments
parser.add_argument('--b', type=str, help='Base dataset')
parser.add_argument('--r', type=str, help='Reflection dataset')
parser.add_argument('--j', type=str, help='Joined dataset')

args = parser.parse_args()

base = args.b
reflection = args.r
data = args.j

files_b = [f for f in os.listdir(base) if os.path.isfile(os.path.join(base, f))]
files_r = [f for f in os.listdir(reflection) if os.path.isfile(os.path.join(reflection, f))]

s = superimpose()
for fb in files_b:
    for fr in files_r:
        img = s.gen(cv2.imread(base + fb), cv2.imread(reflection + fr), 0.3)
        cv2.imwrite(data + os.path.basename(fb).split(".")[0] + "_" + os.path.basename(fr).split(".")[0] + ".png", img)


