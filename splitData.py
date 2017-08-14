#!/usr/local/bin/python

import os
import sys
import numpy as np
import argparse
import errno    
import os, random

def fatal_error(msg):
    print("ERROR")
    print(msg)
    exit(-1)

def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise

def validate_arguments(args):
    if (args.input is None):
        fatal_error('No input path provided.')
    if (args.output is None):
        fatal_error('No output path provided.')
    if (args.validation is None):
        fatal_error('No validation percentage provided.')

def convert(input, output, validation_percentage):
    for folder in os.listdir(input):
        mkdir_p(os.path.join(output,folder))
        subdir=os.path.join(input,folder)
        if os.path.isfile(subdir):
            continue
        numberOfItems=len([name for name in os.listdir(subdir)])
        numbVal = int(numberOfItems * validation_percentage)

        while numbVal>0:
            file = random.choice(os.listdir(os.path.join(input,folder)))
            os.rename(os.path.join(input,folder,file), os.path.join(output,folder,file))
            numbVal-=1

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', help='input path')
    parser.add_argument('--output', help='output path')
    parser.add_argument('--validation', help='validation percentage')
    args = parser.parse_args()
    validate_arguments(args)
    convert(args.input, args.output, float(args.validation))

if __name__ == '__main__':
    main()
