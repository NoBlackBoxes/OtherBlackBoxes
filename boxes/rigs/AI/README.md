# Rigs : AI

Parts lists for a modern AI workstation to run realtime diffusion models.

Example: [Dall-E 2](https://openai.com/dall-e-2/)
Example: [Dream Fusion](https://dreamfusion3d.github.io/index.html)

<hr>

## Hardware

GPU: Nvidia RTX 4090
https://www.overclockers.co.uk/gigabyte-geforce-rtx-4090-gaming-oc-24gb-gddr6x-pci-express-graphics-card-gx-1fc-gi.html

Case: Kolink Void X ARGB Midi Tower Case - Black Window
https://www.overclockers.co.uk/kolink-void-x-argb-midi-tower-case-black-window-ca-05q-kk.html

Power Supply: Gigabyte UD1000GM PG5 1000W 80 PLUS Gold Modular ATX Power Supply
https://www.overclockers.co.uk/gigabyte-ud1000gm-pg5-1000w-80-plus-gold-modular-atx-power-supply-ca-03f-gi.html

Motherboard: Gigabyte Z790 Aero G (LGA 1700) DDR5 ATX Motherboard
https://www.overclockers.co.uk/gigabyte-z790-aero-g-lga-1700-ddr5-atx-motherboard-mb-5bl-gi.html

CPU: Intel Core i9-13900K (Raptor Lake) Socket LGA1700 Processor - Retail
https://www.overclockers.co.uk/intel-core-i9-13900k-raptor-lake-socket-lga1700-processor-retail-cp-6az-in.html

RAM (system memory): Kingston FURY Beast 32GB (2x16GB) DDR5 PC5-44800C40 5600MHz Dual Channel Kit
https://www.overclockers.co.uk/kingston-fury-beast-32gb-2x16gb-ddr5-pc5-44800c40-5600mhz-dual-channel-kit-my-29h-ks.html

Storage (SSD): Samsung 980 Pro 1TB M.2 2280 PCI-e 4.0 x4 NVMe Solid State Drive
https://www.overclockers.co.uk/samsung-980-pro-1tb-m.2-2280-pci-e-4.0-x4-nvme-solid-state-drive-hd-248-sa.html

Cooling (AIO Water)): Gigabyte AORUS WATERFORCE X 240 ARGB Liquid AIO Performance CPU Cooler - 240mm
https://www.overclockers.co.uk/gigabyte-aorus-waterforce-x-240-argb-liquid-aio-performance-cpu-cooler-240mm-hs-00d-gi.html

## Software

Intsall Archlinux

1. Download latest ISO: [Arch Downloads](https://archlinux.org/download/)
2. Write to USB flash drive: Use [Etcher](https://www.balena.io/etcher/)
3. Boot the USB drive: Boots to root user command line
4. Change keyboard layout if necessary (default: US)

    ```bash
    # List all keymaps
    localectl list-keymaps

    # Load correct keymap
    loadkeys <your keymap>
    ```

5. Connect to the internet

    ```bash
    # For WiFi
    iwctl --passphrase passphrase station device connect SSID

    # Wired connections should just work (via DHCP)
    ```

6. Update clock using network time protocol

    ```bash
    timedatectl set-ntp true
    ```

7. Partition disks: ***This will delete the entire drive***

    ```bash
    # List all disks
    fdisk -l

    # Select your drive
    fdisk <your drive> # e.g. /dev/sda
    ```

    - *(From within fdisk command prompt)*
    - Delete all exisiting partitions with command "d"
    - Create ESP parition (for UEFI systems)
      - Create new partion with command "n"
      - Partition number: 1 (default)
      - First sector: 2048 (default)
      - Last sector: +512M
      - (Remove signature, if requested)
    - Change partition type to 'EFI System'
      - Change type with command "t"
      - (Selects partition 1)
      - Partition type or alias: 1
    - Create root partition
      - Create new partion with command "n"
      - Partition number: 2 (default)
      - First sector: ?? (choose default)
      - Last sector: ?? (choose default)
      - (Remove signature, if requested)
    - Write partitioning commands with command "w"

8. Create file systems on the new partitions
  
    ```bash
    # Create FAT32 file system on EFI partition
    mkfs.fat -F32 /dev/[your drive - parttion 1]p1

    # Create EXT4 file system on ROOT partition
    mkfs.ext4 /dev/[your drive - parttion 2]p2
    ```

9. Select download mirror site

    ```bash
    pacman -Syy # (sync)
    pacman -S reflector # Install reflector

    # Rate mirrors and update list
    reflector --verbose --latest 25 --sort rate --save /etc/pacman.d/mirrorlist
    ```

10. Mount the file systems

    ```bash
    mount /dev/<your drive - parttion 2> /mnt
    mkdir /mnt/efi
    mount /dev/<your drive - parttion 1> /mnt/efi
    ```

11. Install essential packages

    ```bash
    pacstrap /mnt base linux linux-firmware vim sudo
    ```

12. Generate FSTAB

    ```bash
    genfstab -U /mnt >> /mnt/etc/fstab
    ```

13. Configure system in Chroot

    ```bash
    # Change root
    arch-chroot /mnt
    
    # Set timezone
    ln -sf /usr/share/zoneinfo/Europe/London /etc/localtime

    # Run hwclock(8) to generate /etc/adjtime:
    hwclock --systohc

    # Edit locale.gen
    vim /etc/locale.gen
    # - uncomment en_GB.UTF-8 UTF-8

    # Create locale.conf
    vim /etc/locale.conf
    # - add LANG=en_US.UTF-8

    # Generate locales
    locale-gen

    # Create the hostname file
    vim /etc/hostname
    # - add <your hostname>

    # Edit host file
    vim /etc/hosts
    # - add 
    #   127.0.0.1	localhost
    #   ::1		    localhost
    #   127.0.1.1	<your hostname>

    # Set up root passwd
    passwd

    # Install microcode (updates for specific CPUs)
    pacman -S amd-ucode # or intel-ucode

    # Install EFI bootloader
    pacman -S grub efibootmgr

    # Create the directory where EFI partition will be mounted:
    mkdir /boot/efi

    # Mount the ESP partition you had created
    mount /dev/[your drive]p1 /boot/efi

    # Install grub and configure
    grub-install --target=x86_64-efi --bootloader-id=GRUB --efi-directory=/boot/efi
    grub-mkconfig -o /boot/grub/grub.cfg

    # Create user
    useradd -m kampff
    passwd kampff

    # Add user to /etc/sudoers
    vim /etc/sudoers
    # - add <USER> ALL=(ALL) ALL

    # Install Xorg
    pacman -S xorg

    # Install KDE (plasma)
    pacman -S plasma plasma-wayland-session
    
    # Install KDE (applications) ???? - Should be more minimal (Netowrk, firefox)
    pacman -S kde-system-meta
    pacman -S kde-utilities-meta
    pacman -S kde-graphics-meta
    pacman -S kde-multimedia-meta

    # Install development tools
    pacman -S base-devel
    pacman -S valgrind
    pacman -S gcc-fortran

    # Install required packages for VK
    pacman -S ffmpeg
    pacman -S minicom
    pacman -S libusb
    pacman -S ncurses
    pacman -S rsync
    pacman -S cpio
    pacman -S wget
    pacman -S bc
    pacman -S dtc
    pacman -S mtools

    # Install firefox
    pacman -S firefox

    # Install utilities
    pacman -S git
    pacman -S usbutils
    # pacman -S code - need Mirosoft's extensions...use their binary version
    
    # Enable display and network manager
    systemctl enable sddm.service
    systemctl enable NetworkManager.service

    # Exit chroot
    exit
    ```

14. Shutdown, remove USB, hope for the best!























