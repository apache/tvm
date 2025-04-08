#!/bin/bash  
 
TEST_DIR="./test_search"  
  
if [ ! -d "$TEST_DIR" ]; then  
  echo "test directory $TEST_DIR doesn't exist!"  
  exit 1  
fi  

echo "Start testing..."  
pytest "$TEST_DIR"  


if [ $? -eq 0 ]; then  
  echo "All test cases pass！"  
else  
  echo "Some test cases fail!"  
  exit 1  
fi
