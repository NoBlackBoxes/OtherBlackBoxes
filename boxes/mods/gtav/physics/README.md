# Mods :: GTAV :: Physics

How to mod game physics? What is game physics?

## Install OpenGL (for Python)

- Make sure python is installed (and pip)

```bash
python --version
python -m ensurepip
```

- (Windows) pyopengl and pyopengl_accelerate
  - Download the correct versions from here (included GLUT): https://www.lfd.uci.edu/~gohlke/pythonlibs/#pyopengl
  - Navigate to download folder...

```bash
python -m pip install PyOpenGL-3.1.5-cp310-cp310-win_amd64.whl
python -m pip install PyOpenGL_accelerate-3.1.5-cp310-cp310-win_amd64.whl
```

- We will also use the keyboard module in the examples

```bash
python -m pip install keyboard
```