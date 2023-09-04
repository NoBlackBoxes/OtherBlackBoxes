# Mods :: GTAV :: C++

Modding GTA V (and earlier versions) relies on DLL injection to "hook" custom mod scripts (*.asi files) into the running game. ASI files are simply renamed DLLs, which are usually compiled using MSVC and written in C++.

## Install and Configure VS Code

- Download and install VS code
- Install C/C++ extension
- Install the Microsoft Visual C++ (MSVC) compiler toolset (https://visualstudio.microsoft.com/downloads/#other)
  - Build Tools for Visual Studio 2022
- Test your GCC C++ compiler

## Setup DLL injection

- Download the latest version of Script Hook V (http://www.dev-c.com/gtav/scripthookv/)
  - And the SDK
- Extract both
- Copy the contents of the "bin" folder to your GTA root directory
- Add "ScriptHookV.dev" file to GTA root directory (for reloading/unloading scripts with *Ctl+R*)
- Copy the inc/ folder contents in the SDK to a folder named external/include in the "gtav" repo
- Copy the lib/ folder contents in the SDK to a folder named external/lib in the "gtav" repo
- Copy the scripts.h and main.h header from header to the "external/includes" folder
- Change the include paths of the scripts.h header (remove "../../inc")

## Build a mod (*.asi)
- Open the "x64 native tools command prompt" (to access the MSVC C++ compiler)
- Navigate to the source folder
