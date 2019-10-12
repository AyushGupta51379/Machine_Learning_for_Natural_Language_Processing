#!/bin/bash
#PBS -N lab8
#PBS -e ./error_log.txt
#PBS -o ./outptu_log.txt

cd ~/lab8
echo Start of calculation
python lab8_answer.py
echo End of calculation
