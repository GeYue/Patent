#!/bin/bash

model="Berta-PTL"
version="v8"

for fold in {0..6}; do
        #time zip -r ./${model}-fold-${fold}-${version}-BV.zip ./*fold-${fold}-BV.pth
        time zip -r ./${model}-fold-${fold}-${version}-BS.zip ./*fold-${fold}-BS.pth
done
