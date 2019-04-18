#!/bin/bash

M=("web-Google" "2cubes_sphere" "cage12" "cage15" "wiki-Talk")
URLs=(
    "https://sparse.tamu.edu/MM/SNAP/web-Google.tar.gz"
    "https://www.cise.ufl.edu/research/sparse/MM/Um/2cubes_sphere.tar.gz"
    "https://www.cise.ufl.edu/research/sparse/MM/vanHeukelum/cage12.tar.gz"
    "https://www.cise.ufl.edu/research/sparse/MM/vanHeukelum/cage15.tar.gz"
    "https://sparse.tamu.edu/MM/SNAP/wiki-Talk.tar.gz"
)


counter=0
for url in ${URLs[@]}; do
    wget $url
    tar -xvzf ${M[$counter]}".tar.gz"
    mv ${M[$counter]}/${M[$counter]}".tar.gz" assets/${M[$counter]}".tar.gz"
    rm -rf ${M[$counter]}*
    ((counter++))
done
