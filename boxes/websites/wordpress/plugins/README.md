# boxes : websites : wordpress : plugins

Creating WordPress Plugins

## Setup Local server, scripting, and database

- See previous

## Duplicate WordPress site locally

- Login to cPanel and compress site files: "public_html_ contents
  - Download and extract at local site location

- Use phpMyAdmin to export wordpress database
-   https://wiki.archlinux.org/title/PhpMyAdmin
-   Go to http://localhost/phpmyadmin/

- Install wp-cli from AUR (https://aur.archlinux.org/wp-cli.git)

```bash
# Import exported database (or use phpMyAdmin)
wp --quiet db import ~/Downloads/cajal_wordpress.sql --dbuser=kampff --dbpass='passwd'

# Make a new wp-config.php
wp config create --dbname=cajal_wordpress --dbuser=kampff --dbpass='passwd'
```

- Edit the URL?

```bash
wp search-replace https://cajal-training.org http://localhost --dry-run --allow-root
wp search-replace https://cajal-training.org http://localhost --allow-root
```

## Install in Sites

