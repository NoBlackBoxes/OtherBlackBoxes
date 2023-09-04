# Build hello_mod

# Set environment variables
set GTAV=D:\SteamLibrary\steamapps\common\Grand Theft Auto V
set ROOT=C:/NoBlackBoxes/OtherBlackBoxes/boxes/mods/gtav
set INCLUDE=%INCLUDE%;%ROOT%/external/include;%ROOT%/external
set LIB=%LIB%;%ROOT%/external/lib

# Compile
cl /LD hello_mod.cpp keyboard.cpp script.cpp ScriptHookV.lib

# Install
xcopy hello_mod.dll "%GTAV%/hello_mod.asi"