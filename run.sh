#!/bin/bash

rm -f ./output.log
rm -f ./pt*.pth
rm -f ./model_initialized.bin
#rm -f ./*.zip

if [ $# -lt 2 ]
then
        printf "Please input 'model type' and 'version' parameters!\n"
	exit 255
fi

model=$1
version="v$2"

if [ "x${model}" = "xr" ]; then
        model="RoBerta-L"
elif [ "x${model}" = "xd" ]; then
        model="DeBerta-V3L"
elif [ "x${model}" = "xx" ]; then
        model="DeBerta-V2XL"
elif [ "x${model}" = "xxx" ]; then
        model="DeBerta-V2XXL"
elif [ "x${model}" = "xb" ]; then
        model="Berta-base"
elif [ "x${model}" = "xbp" ]; then
        model="Berta-PTL"
elif [ "x${model}" = "xbr" ]; then
        model="BART-PTB"
else
        printf "model type error!!!!\n"
        exit 256
fi

printf "$# parameters:: model==${model} version=${version}\n"

python ./Patent_xxBERTa.py $1
if [ $? -ne 0 ]; then
	exit 258
fi

cp ./output.log ./output.log.${version}
#nbme_DeBerta-V2-XLarge_fold-1-BV.pth
for fold in {0..4}; do
	#time zip -r ./${model}-fold-${fold}-${version}-BV.zip ./*fold-${fold}-BV.pth 
	time zip -r ./${model}-fold-${fold}-${version}-BS.zip ./*fold-${fold}-BS.pth 
done


