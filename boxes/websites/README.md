# boxes : websites

## Server

- Install Apache HTTP Server
  - Configuration is at /etc/httpd/conf
  - Enable rewrite_module


## Scripting

- Install PHP (and relevant extensions) server scripting language

- Setup PHP in Apache (change conf)

```bash
sudo pacman -Sy php php-{cgi,gd,pgsql,fpm,apache}
```

Uncomment the following lines in /etc/php/php.ini to enable MySQL/MariaDB extensions:
extension=pdo_mysql
extension=mysqli

#LoadModule mpm_event_module modules/mod_mpm_event.so
LoadModule mpm_prefork_module modules/mod_mpm_prefork.so
Include conf/extra/php_module.conf

## Database

- Install MariaDB database implementation

```bash
mariadb-install-db --user=mysql --basedir=/usr --datadir=/var/lib/mysql
```

- Start service, lging to mariadb, make database

```bash
sudo mysql -u root -p
MariaDB> CREATE DATABASE wp;
MariaDB> GRANT ALL PRIVILEGES ON wp.* TO "wp-user"@"localhost" IDENTIFIED BY "choose_db_password";
MariaDB> FLUSH PRIVILEGES;
MariaDB> EXIT
```

phpmyadmin can help

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