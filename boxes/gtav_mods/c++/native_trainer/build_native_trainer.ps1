# Build hello_asi

# Create output directory
Remove-Item bin -Recurse -ErrorAction Ignore
mkdir bin

# Compile
cl /LD native_trainer.cpp keyboard.cpp script.cpp lib/ScriptHookV.lib
