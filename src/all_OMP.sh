search_dir=./OMP
input_dir=../input

# Set the patterns for the file names you want to select
for input_entry in "$input_dir"/**/*
do
  # Use find to select only files that match the desired pattern
  for entry in "$search_dir"/*/*
  do
      if file "$entry" | grep -q "executable"; then      
# Remove "./" from entry
      entry=${entry#./}
      for t in 1 2 4 8 16
      do
        export OMP_NUM_THREADS=$t
        # Run the command using the selected file
        echo "$entry" "$input_entry" "$t"
      done
      echo " "
    fi
    done
break
done



##ONLY FILES THAT STARTS WITH BLOCK, EBLOCK AND GRID WILL BE SUBMITTED
#search_dir=./input
#for entry in "$search_dir"/*/*
#do
#  # if entry name starts with block or eblock or grid
#  if [[ $entry == ./input/block* ]] || [[ $entry == ./input/eblock* ]] || [[ $entry == ./input/grid* ]]; then
#    #remove ./ from entry
#    entry=${entry:2}
#    for t in 1 2 4 8 16
#    do
##       export OMP_NUM_THREADS=$t
##       echo "./main $entry $t"
#       echo "$entry ./input/cant/cant.mtx $t"
#    done
##    qsub $entry
#  fi
#
##  echo "qsub $entry"
#done


#search_dir=./input
#for entry in "$search_dir"/*/*
#do
#  #if entry name ends with .sub
#  if [[ $entry == *.* ]]; then
#    #remove ./ from entry
#    entry=${entry:2}
#    echo "Submitting $entry"
#    for t in 1 2 4 8 16
#    do
#       export OMP_NUM_THREADS=$t
#       echo "./main $entry $t"
#    done
##    qsub $entry
#  fi
#
##  echo "qsub $entry"
#done
