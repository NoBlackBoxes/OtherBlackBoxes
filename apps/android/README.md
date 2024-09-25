# Apps : Android

Creating an Android app using as little tooling, boilerplate, and mystery dependencies as possible.

## Setup Android development tools

```bash
# Set OtherBlackBoxes repository root environment variable
OBBROOT="/home/$USER/NoBlackBoxes/OtherBlackBoxes"

# Install Open JDK
sudo pacman -S jdk17-openjdk

# Install Android SDK command line tools
cd $OBBROOT/_tmp
mkdir android
```

- Download [Android command line tools](https://developer.android.com/studio#downloads)
  - Extract to $OBBROOT/_tmp/android/tools

```bash
cd $OBBROOT/_tmp/android
unzip commandlinetools-linux-XXXXXXXX_latest.zip
```

```bash
# Install SDK packages
cd $OBBROOT/_tmp/android/cmdline-tools/bin
./sdkmanager --sdk_root=$OBBROOT/_tmp/android "platform-tools"
./sdkmanager --sdk_root=$OBBROOT/_tmp/android "build-tools;34.0.0"
./sdkmanager --sdk_root=$OBBROOT/_tmp/android "platforms;android-34" # Android 14

# Install ADB
sudo pacman -S android-tools

# Create keystore for App signing
keytool -genkeypair -keystore "$OBBROOT/_tmp/android/keystore.jks" -alias androidkey -validity 10000 \
    -keyalg RSA -keysize 2048 -storepass android -keypass android
```

## Build app

- Run the corresponding bash script in the app development folder, e.g. "./build_android_java.sh"

## Install app

- Turn on "developer options" on your phone: [instructions](https://developer.android.com/studio/debug/dev-options)
- Connect phone via USB cable

```bash
# From app folder
adb install -r build/{"APP_NAME}.apk
```