# Build model_switcher

# Set environment variables
set GTAV=C:/Program Files (x86)/Steam/steamapps/common/Grand Theft Auto V
set ROOT=C:/NoBlackBoxes/OtherBlackBoxes/boxes/mods/gtav
set INCLUDE=%INCLUDE%;%ROOT%/external/include;%ROOT%/external
set LIB=%LIB%;%ROOT%/external/lib

# Compile
cl /LD model_switcher.cpp keyboard.cpp script.cpp ScriptHookV.lib

# Install
xcopy model_switcher.dll "%GTAV%/model_switcher.asi"