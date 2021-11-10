# Build hello_asi

# Create output directory
Remove-Item bin -Recurse -ErrorAction Ignore
mkdir bin

# Compile
g++ -c -I . -o bin/asi.o asi.c

# Link
g++ -o bin/asi.asi -s -shared bin/asi.o "-Wl,--subsystem,windows"
