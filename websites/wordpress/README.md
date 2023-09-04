# boxes : websites : wordpress

## Server

- Install Apache HTTP Server (sudo pacman -S apache)
  - Configuration is at /etc/httpd/conf/httpd.conf
  - Enable rewrite_module (uncomment: LoadModule rewrite_module modules/mod_rewrite.so in /etc/httpd/conf/httpd.conf)
  - Change user and group
  - Change DocumentRoot
  - Change to "AllowOverride All"
  - Uncomment and change to "ServerName localhost:80"
  - Edit /etc/hosts
  - Start httpd service

## Scripting

- Install PHP (and relevant extensions) server scripting language

```bash
sudo pacman -Sy php php-{cgi,gd,pgsql,fpm,apache}
```

- Uncomment the following lines in /etc/php/php.ini to enable MySQL/MariaDB extensions:
```bash
extension=pdo_mysql
extension=mysqli
extension=iconv
```

- Turn on display errors! (it helps debug)

- Setup PHP in Apache (change conf)
```bash

# In /etc/httpd/conf/httpd.conf uncomment/add...

#LoadModule mpm_event_module modules/mod_mpm_event.so
LoadModule mpm_prefork_module modules/mod_mpm_prefork.so
Include conf/extra/php_module.conf

# In /etc/httpd/conf/httpd.conf add...
# After LoadModules
LoadModule php_module modules/libphp.so
AddHandler php-script .php

# At the end
<IfModule dir_module>
    DirectoryIndex index.php
</IfModule>
```

## Database

- Install MariaDB database implementation (sudo pacman -S mariadb)

```bash
sudo mariadb-install-db --user=mysql --basedir=/usr --datadir=/var/lib/mysql
```

- Start service, login to mariadb, make database

```bash
sudo mysql -u root -p
MariaDB> CREATE USER 'kampff'@'localhost' IDENTIFIED BY 'some_pass';
MariaDB> GRANT ALL PRIVILEGES ON mydb.* TO 'kampff'@'localhost';
MariaDB> FLUSH PRIVILEGES;
MariaDB> CREATE DATABASE cajal_wordpress;
MariaDB> GRANT ALL PRIVILEGES ON cajal_wordpress.* TO "kampff"@"localhost" IDENTIFIED BY "choose_db_password";
MariaDB> FLUSH PRIVILEGES;
MariaDB> EXIT
```

- Install phpMyAdmin (sudo pacman -S phpmyadmin)

```bash
# Create /etc/httpd/conf/extra/phpmyadmin.conf with...
Alias /phpmyadmin "/usr/share/webapps/phpMyAdmin"
<Directory "/usr/share/webapps/phpMyAdmin">
    DirectoryIndex index.php
    AllowOverride All
    Options FollowSymlinks
    Require all granted
</Directory>

# In /etc/httpd/conf/httpd.conf add...
Include conf/extra/phpmyadmin.conf
```

## CMS

- Download latest wordpress version

```bash
cd /srv/http/<name of site>
wget https://wordpress.org/latest.tar.gz
tar xvzf latest.tar.gz
```

## Setup wordpress

- Create wordpress 
- permissions (server user vs wordpress)