#!/bin/bash

# Build and package an Android App
set -eu

# Set Voight-Kampff Root
OBBROOT="/home/kampff/NoBlackBoxes/OtherBlackBoxes"
OBBTMP=$OBBROOT"/_tmp"

# Export App name
APP_NAME="hello_java"

# Export environment variables
ANDROID_BUILD_TOOLS="${OBBTMP}/android/build-tools/34.0.0"
ANDROID_PLATFORM="${OBBTMP}/android/platforms/android-34"
#JAVA_JRE="/usr/lib/jvm/java-17-openjdk/jre/lib/rt.jre"
ANDROID_NDK="${OBBTMP}/android/ndk-bundle"
ANDROID_ARM_TOOLCHAIN="${OBBTMP}/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android30-clang"

# Create build output folders
mkdir -p build/gen 
mkdir -p build/obj
mkdir -p build/apk

# AAPT: Generate R.java file
"${ANDROID_BUILD_TOOLS}/aapt" package -f -m -J build/gen/ -S res -M AndroidManifest.xml -I "${ANDROID_PLATFORM}/android.jar"
echo "R.java generated"

# Compile Java
javac -source 17 -target 17 \
    -classpath "${ANDROID_PLATFORM}/android.jar" -d build/obj \
    build/gen/vk/nbb/${APP_NAME}/R.java java/vk/nbb/${APP_NAME}/MainActivity.java
echo "Java compiled"

# Convert to Dalvik
${ANDROID_BUILD_TOOLS}/d8 --no-desugaring --output build/apk build/obj/vk/nbb/${APP_NAME}/*.class
echo "Converted to Dalvik"

# Package into APK
"${ANDROID_BUILD_TOOLS}/aapt" package -f -M AndroidManifest.xml -S res/ \
    -I "${ANDROID_PLATFORM}/android.jar" \
    -F build/${APP_NAME}.unsigned.apk build/apk/
echo "Packaged APK"

# Align on 4-byte boundaries
"${ANDROID_BUILD_TOOLS}/zipalign" -f -p 4 build/${APP_NAME}.unsigned.apk build/${APP_NAME}.aligned.apk
echo "Aligned APK"

# Sign App with pre-made key (keystore.jks)
"${ANDROID_BUILD_TOOLS}/apksigner" sign --ks "$OBBTMP/android/keystore.jks" \
    --ks-key-alias androidkey --ks-pass pass:android \
    --key-pass pass:android --out build/${APP_NAME}.apk \
    build/${APP_NAME}.aligned.apk
echo "Signed APK"

# Install Application
adb install -r "build/${APP_NAME}.apk"
echo "Installed APK"

#FIN