#!/bin/bash

for job in $(seq 143289 143619):
do
qdel $job
done 
done
