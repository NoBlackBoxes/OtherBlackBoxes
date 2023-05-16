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
rsync -rL /home/ubuntu/NoBlackBoxes/repos/OtherBlackBoxes/boxes/websites/applications/flask/gptvswiki /var/www/llm
rsync -rL /home/ubuntu/NoBlackBoxes/repos/OtherBlackBoxes/boxes/websites/applications/flask/wsgi/llm.wsgi /var/www/llm
```

- Copy the default config file in "/etc/apache2/sites-available" to one named after your site, edit appropriately

```text
<VirtualHost *>
    ServerName noblackboxes.org

    WSGIDaemonProcess llm user=ubuntu group=ubuntu threads=4
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

- Change owner of site folder to user/group mentioned in Apache config (!): chmod -R 464

## Encryption (HTTPS)

Use Let's Encrypt's certbot

- Modify Apache conf file to use both ports 80 (http) and 443 (https)

```text
WSGIDaemonProcess llm user=ubuntu group=ubuntu threads=4
WSGIScriptAlias / /var/www/llm/llm.wsgi

<VirtualHost *:80>
    ServerName noblackboxes.org

    <Directory /var/www/llm>
        WSGIProcessGroup llm
        WSGIApplicationGroup %{GLOBAL}
        Options FollowSymLinks
        AllowOverride None
        Require all granted
    </Directory>
</VirtualHost>

<VirtualHost *:443>
    ServerName noblackboxes.org

    <Directory /var/www/llm>
        WSGIProcessGroup llm
        WSGIApplicationGroup %{GLOBAL}
        Options FollowSymLinks
        AllowOverride None
        Require all granted
    </Directory>
</VirtualHost>
```


```bash
sudo pip install certbot certbot-apache
sudo certbot --apache
```

- Make sure port is open (443) on AWS


- to restart server
```bash
sudo systemctl restart apache2
```