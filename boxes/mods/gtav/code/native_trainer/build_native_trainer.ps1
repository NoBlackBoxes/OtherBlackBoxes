# Build native_trainer

# Set environment variables
set GTAV=C:/Program Files (x86)/Steam/steamapps/common/Grand Theft Auto V
set ROOT=C:/NoBlackBoxes/OtherBlackBoxes/boxes/mods/gtav
set INCLUDE=%INCLUDE%;%ROOT%/external/include;%ROOT%/external
set LIB=%LIB%;%ROOT%/external/lib

# Compile
cl /LD native_trainer.cpp keyboard.cpp script.cpp ScriptHookV.lib

# Install
xcopy native_trainer.dll "%GTAV%/native_trainer.asi"