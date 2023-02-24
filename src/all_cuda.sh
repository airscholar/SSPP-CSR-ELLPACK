search_dir=./CUDA
input_dir=../input

# Set the patterns for the file names you want to select
for input_entry in "$input_dir"/**/*; do
  # Use find to select only files that match the desired pattern
  for entry in "$search_dir"/*/*; do
    # If the file is an executable
    if file "$entry" | grep -q "executable"; then
      # Remove "./" from entry
      entry=${entry#./}
        # Run the command using the selected file
        echo "$entry" "$input_entry" "$t"
      echo " "
    fi
  done
done

