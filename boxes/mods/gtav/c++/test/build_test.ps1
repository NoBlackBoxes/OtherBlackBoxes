# Build hello_test

# Create output directory
Remove-Item bin -Recurse -ErrorAction Ignore
mkdir bin

# Compile
g++ -c -I . -o bin/test.o test.c

# Link
g++ -o bin/test.asi -s -shared bin/test.o "-Wl,--subsystem,windows"
