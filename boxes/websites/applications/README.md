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

## Self-signed SSL certificates (for testing)

- in /etc/httpd/conf

```bash
# Generate
sudo openssl genrsa -des3 -out server.key 1024

# Self-sign (use correct common name : domain name or IP)
sudo openssl req -new -key server.key -out server.csr

# Remove passphrase
sudo cp server.key server.key.org
sudo openssl rsa -in server.key.org -out server.key

# Specify expiration
sudo openssl x509 -req -days 365 -in server.csr -signkey server.key -out server.crt

# Modify conf
sudo nano /etc/httpd/conf/httpd.conf

##Once there, uncomment the following line:
## Include conf/extra/httpd-ssl.conf

# Restart apache
sudo systemctl restart httpd

# May need to turn on some Apche Modules
# LoadModule ssl_module modules/mod_ssl.so
# LoadModule socache_shmcb_module modules/mod_socache_shmcb.so
```