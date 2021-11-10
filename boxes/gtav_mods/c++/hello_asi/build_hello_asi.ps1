# Build hello_asi

# Create output directory
Remove-Item bin -Recurse -ErrorAction Ignore
mkdir bin

# Compile
cl /LD hello_asi.cpp keyboard.cpp script.cpp lib/ScriptHookV.lib
