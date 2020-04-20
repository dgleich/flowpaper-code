#!/bin/bash
FILES=`ls *.pdf`
for file in $FILES; do 
  echo pdf2ps $file
  echo ps2pdf "${file%.*}".ps
done  
