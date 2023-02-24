search_dir=./CUDA
input_dir=./input
for input_entry in "$input_dir"/**/*
do
  for entry in "$search_dir"/**/*
  do
    # if entry name starts with block or eblock or grid
    if [[ $entry == ./CUDA/CSR/CSR_* ]] || [[ $entry == ./CUDA/ELLPACK/EBlock* ]] && [[ $entry != *.cu ]]; then
      #remove ./ from entry
      entry=${entry:2}
      for t in 1 2 4 8 16
      do
  #       export OMP_NUM_THREADS=$t
  #       echo "./main $entry $t"
         ". ./$entry $input_entry $t"
      done
      echo " "
    fi
  done
break
done

