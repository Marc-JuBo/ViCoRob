#!/bin/sh
# Intentarem re-anomenar les etiquetes, dels valors dolents als valors bons.

# ----------------------- SCRIPT 5 ---------------------------
 
# Assegurem que el fitxer destí estigui buit.
if [ -f merged_modified.csv ];
  then
  	rm merged_modified.csv
fi

cp merged.csv merged_modified.csv

for j in {0..20} 
	 do
		sed -i 's/^'$j',/'$j'0000,/' "./merged_modified.csv"
	 done

declare -a old_ids[25]
declare -a new_ids[25]
COUNTER=0
first=1
while read -r line 
 do
	IFS=',' read -ra id_pair <<< $line
	if [ -z "${id_pair[0]}" ] 
	then :
	else
		old_ids[$COUNTER]=${id_pair[0]}
		new_ids[$COUNTER]=${id_pair[1]}
	fi
	
	# This line will be useful to check if everything is working until here.
	#echo "${old_ids[COUNTER]} ${new_ids[COUNTER]}"
	(( COUNTER++ ))

 done <<< $(cat "./merged_modified.csv")

lenght=${#old_ids[@]}

# ----------------------- SCRIPT 2 ---------------------------
path=$(pwd)
for d in $(ls); 
do
  if [[ -d $d ]];
  then
  	echo "$path/$d/labels/"
	cd $path/$d/labels/
	for i in $(ls); 
	 do
		for j in {0..20};
		 do
			sed -i 's/^'$j' /'$j'0000 /' ./"$i"
		 done
		 
	# ----------------------- SCRIPT 6 ---------------------------
		for k in {1..20}
		 do
		 	sed -i 's/^'${old_ids[k]}' /'${new_ids[k]}' /' ./"$i"
		 done
	 done
  fi
  cd $path
done

