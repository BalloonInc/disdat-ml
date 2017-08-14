#!/bin/bash

# removes all files with the extension JPG, jpg, JPEG, jpeg that are not actual JPEG files.
numberOfDeletedFiles=0
find data/. -type f -iname '*.jpeg' -o -iname '*.jpg' -exec bash -c 'file -b "$1" | grep -qi jpeg || (echo "removing invalid file $1" && rm "$1" )' none {} \;
