# boxes : websites : applications

Larger scale websites

## The LAMP stack

- Install Apache HTTP server

    ```bash
    sudo apt-get install apache2
    ```

- Install WSGI python interface and enable module for Apache

    ```bash
    sudo apt-get install libapache2-mod-wsgi-py3
    sudo a2enmod wsgi
    ```

- Use default Apache folder

- Copy app to __init__.py

```bash
cp /home/ubuntu/NoBlackBoxes/repos/OtherBlackBoxes/boxes/websites/applications/flask/gptvswiki/app.py /home/ubuntu/NoBlackBoxes/repos/OtherBlackBoxes/boxes/websites/applications/flask/gptvswiki/__init__.py
```

- Sync project folder

```bash
sudo rsync -rL /home/ubuntu/NoBlackBoxes/repos/OtherBlackBoxes/boxes/websites/applications/flask/gptvswiki/* /var/www/llm
```

