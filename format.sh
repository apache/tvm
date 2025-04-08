#!/bin/bash  

# Check if a directory argument is provided  
if [ -z "$1" ]; then  
  echo "Please provide a directory to process."  
  echo "Usage: $0 <directory>"  
  exit 1  
fi  

# Function to process a Python file  
process_file() {  
  local file="$1"  
  echo "Processing file: $file"  

  # Remove unused imports  
  autoflake --remove-all-unused-imports --remove-unused-variables --in-place "$file"  

  # Sort import modules  
  isort "$file"  

  # Format the code using black  
  black --line-length 79 "$file" 

  autopep8 --in-place --aggressive --max-line-length 79 "$file"

  flake8 "$file" 
}


export -f process_file  # Export function for parallel use  

# Find all Python files and process them in parallel  
find "$1" -type f -name "*.py" | parallel process_file  

echo "Unused imports removed and all Python files formatted."
