search_dir=./input
for entry in "$search_dir"/*/*
do
  #if entry name ends with .sub
  if [[ $entry == *.* ]]; then
    #remove ./ from entry
    entry=${entry:2}
    echo "Submitting $entry"
    for t in 1 2 4 8 16
    do
       export OMP_NUM_THREADS=$t
       echo "./main $entry $t"
    done
#    qsub $entry
  fi

#  echo "qsub $entry"
done