# Mods :: GTAV

Modding GTA V (and earlier versions) relies on DLL injection to "hook" custom mod scripts (*.asi files) into the running game. ASI files are simply renamed DLLs, which are usinally compiled using MSVC and written in C++.

## Install and Configure VS Code

- Download and install VS code
- Install C/C++ extension
- Install Powershell extension
- Install the GCC compiler (https://www.msys2.org/): MinGW
  - Follow the install instructions on the website
- Install the Microsoft Visual C++ (MSVC) compiler toolset (https://visualstudio.microsoft.com/downloads/#other)
  - Build Tools for Visual Studio 2022
- Test your GCC C++ compiler
  - Add path (settings...environment variables...path..user and system)
  - Reload

## Setup DLL injection

- Download the latest version of Script Hook V (http://www.dev-c.com/gtav/scripthookv/)
  - And the SDK
- Extract
- Copy the contents of the "bin" folder to your GTA root directory
- Add "ScriptHookV.dev" file to main folder (for reloading/unloading scripts with *Ctl+R*)

