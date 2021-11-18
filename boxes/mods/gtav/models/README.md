# Mods :: GTAV :: Models

How do I add new (or replace) models in GTA V?

## OpenIV

OpenIV is an application that can extract, view, and edit GTA V archives containing game assets. Game assets are all textures and 3D models that are used by the game engine to create the visuals of GTA.

- Download and install OpenIV: https://openiv.com/
- Run OpenIV (select GTA V for windows from the start page)
- Choose your GTA V install directory
- Create a "mods" folder in your GTAV root game folder
- Copy your original game asset files (all .rpf files)
  - common.rpf
  - x64*.rpf
  - update folder
  - x64 folder
- Select "ASI manager" from thre tools menu and install OpenIV.asi
  - This ASI script will force GTA V to use the assets in your mod folder rather than the originals. Remove (uninstall) this script to play the original game.

### Modify an exisiting texture

We will change the texture of the blimp.

- Navigate to this archive within OpenIV:
  - GTAV\mods\x64w.rpf\dlcpacks\spupgrade\dlc.rpf\x64\levels\gta5\vehicles\upgradevehicles.rpf
- Export the blimp2.ytd and blimp2+hi.ytd as PNG files)
- Edit as you see fit (yet keep the original size)
- Turn on "edit mode" in OpenIV
- Select the blimp textures, and choose replace, replace with your modified versions.
- Start GTA V and spawn a blimp

### Modify an exisiting model

Again, we will alter the model of the blimp.
- Navigate to this archive within OpenIV:
  - GTAV\mods\x64w.rpf\dlcpacks\spupgrade\dlc.rpf\x64\levels\gta5\vehicles\upgradevehicles.rpf
- Export it somewhere
- Download Zmodeler: https://www.zmodeler3.com/
- Get license/account
- Open YFT
- Change
- Export
- Replace in game (same name)
