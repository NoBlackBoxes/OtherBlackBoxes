# boxes : websites : wordpress : plugins

Creating WordPress Plugins

## Setup Local server and database

- Install Apache and configure - start httpd.service
- Install PHP and configure
- Install MariaDB - start mariadb.service
  - Create new user/password/database

    ```bash
    sudo mysql -u root -p
    MariaDB> CREATE USER 'kampff'@'localhost' IDENTIFIED BY 'some_pass';
    MariaDB> GRANT ALL PRIVILEGES ON mydb.* TO 'kampff'@'localhost';
    MariaDB> FLUSH PRIVILEGES;
    MariaDB> CREATE DATABASE wordpress;
    MariaDB> GRANT ALL PRIVILEGES ON wordpress.* TO 'kampff'@'localhost' IDENTIFIED BY 'some_pass';
    MariaDB> FLUSH PRIVILEGES;
    MariaDB> EXIT
    ```

## Duplicate WordPress site locally

- Login to cPanel and compress site files: "public_html_ contents
  - Download and extract at local site location

- Use phpMyAdmin to export wordpress database
-   https://wiki.archlinux.org/title/PhpMyAdmin
-   Go to http://localhost/phpmyadmin/


```bash
# Make a new wp-config.php
wp config create --dbname=cajal_wordpress --dbuser=kampff --dbpass='passwd'

# Import exported database (or use phpMyAdmin)
wp --quiet db import ~/Downloads/cajal_wordpress.sql
```

- Edit the URL?

```bash
wp search-replace https://cajal-training.org http://localhost --dry-run --allow-root
wp search-replace https://cajal-training.org http://localhost --allow-root
```

## Install in Sites

