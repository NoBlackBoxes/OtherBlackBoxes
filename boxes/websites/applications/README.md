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

- Use default Apache folder (/var/www/<site-name>)

- Sync project folder(s)

```bash
sudo rsync -rL /home/ubuntu/NoBlackBoxes/repos/OtherBlackBoxes/boxes/websites/applications/flask/gptvswiki /var/www/llm
sudo rsync -rL /home/ubuntu/NoBlackBoxes/repos/OtherBlackBoxes/boxes/websites/applications/flask/wsgi/llm.wsgi /var/www/llm
```

- Copy the default config file in "/etc/apache2/sites-available" to one named after your site, edit appropriately

```text
<VirtualHost *>
    ServerName noblackboxes.org

    WSGIDaemonProcess llm 
    WSGIScriptAlias / /var/www/llm/llm.wsgi

    <Directory /var/www/llm>
        WSGIProcessGroup llm
        WSGIApplicationGroup %{GLOBAL}
        Order deny,allow
        Allow from all
    </Directory>
</VirtualHost>
```

- Activate site, restart Apache

```bash
sudo a2ensite llm
sudo a2dissite 000-default.conf
```


- CHange owner of site folder to user/group mentione din APache config