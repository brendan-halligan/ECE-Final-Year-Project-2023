from string import ascii_lowercase
from random import choice, randint
import os


def randomize_files(dir):
    for f in os.listdir(dir):
        path = os.path.join(dir, f)
        if os.path.isfile(path):
            newpath = os.path.join(dir, ''.join([choice(ascii_lowercase) for _ in range(randint(6, 10))]))
            os.rename(path, newpath + ".png")


randomize_files("extracted_frames")
