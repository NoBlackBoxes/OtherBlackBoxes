# GTAV Mods

How? What's going on?

## Install and Configure VS Code

- Download and install VS code
- Install C/C++ extension
- Install Powershell extension
- Install a compiler (https://www.msys2.org/): MinGW
  - Follow the install instructions on the website
  - Install the Microsoft Visual C++ (MSVC) compiler toolset (https://visualstudio.microsoft.com/downloads/#other)
    - Build Tools for Visual Studio 2022

- Test your new C++ compiler
  - Add path (settings...environment variables...path..user and system)
  - Reload

## Setup DLL injection

- Download the latest version of Script Hook V (http://www.dev-c.com/gtav/scripthookv/)
  - And the SDK
- Extract
- Copy the contents of the "bin" folder to your GTA root directory

- Add "ScriptHookV.dev" file to main folder (fpr reloading/unloading scripts with ctl+R)

