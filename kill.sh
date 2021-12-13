#!/bin/bash

for job in $(seq 1312797 1312812):
do
qdel $job
done 
