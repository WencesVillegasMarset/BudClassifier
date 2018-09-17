#!/bin/bash

dirInput=$1
dfiles="${dirInput}*.txt"

dirOutput="${dirInput}analisys/"
mkdir $dirOutput

summaryOut="${dirOutput}summary.csv"

echo "Dict,R,It,Accuracy,Prec-TRUE,Rec-TRUE,F-Scr-TRUE,Prec-FALSE,Rec-FALSE,F-Scr-FALSE" > $summaryOut

for archivo in $dfiles; do
	measures=`awk 'NR==12' $archivo | cut -c 6-77`
	datasets=`awk 'NR==2' $archivo | cut -c 27-38`
	output=`echo $datasets $measures`
	output=${output//' '/','}
	echo ${output//'_'/','} >> $summaryOut
done

R --vanilla --slave --args ${dirOutput} < process_res_est.R