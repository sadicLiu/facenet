#!/bin/bash

getFormatIndex() {
    if [ $1 -lt 10 ]
    then
        echo "00$1"
    elif [ $1 -lt 100 ]
    then
        echo "0$1"
    else
        echo $1
    fi
}

read -p "Source folder: " src
read -p "Target folder: " target
read -p "Class start index: " start
read -p "Class end index: " end
read -p "Step size: " step

mkdir -p ${target}

num_cls=$(ls -l ${src} | wc -l)
echo There are ${num_cls} classes.
echo Start processing...

for cls in $(seq ${start} ${end})
do
    mkdir ${target}/${cls}

    num_src_img=$(ls -l ${src}/${cls} | wc -l)
    let num_tar_img=${num_src_img}/${step}
    echo ${num_tar_img} images picked for class ${cls}

    for img in $(seq 1 ${num_tar_img})
    do
        let src_idx=${img}-1
        let src_idx=${src_idx}*${step}
        let src_idx++

        src_idx=`getFormatIndex ${src_idx}`
        tar_idx=`getFormatIndex ${img}`
        src_idx=$(echo ${src_idx})
        tar_idx=$(echo ${tar_idx})

        src_img_path=${src}/${cls}/${cls}_frame${src_idx}.jpg
        tar_img_path=${target}/${cls}/${cls}_frame${tar_idx}.jpg

        cp ${src_img_path} ${tar_img_path}
    done
done

echo Success.



