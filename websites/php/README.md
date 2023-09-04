# boxes : websites : php

## Setup

- Install Apache HTTP Server
- Install PHP
- Install a database

## Simple

- Create a simple PHP echo page (embedded in HTML)

```html
<html>
<head>
<title>PHP Hello</title>
</head>
<body>
<?php echo '<p>Hello World</p>'; ?>
</body>
</html>
```

- Create a simple HTML form

```html
<html>
<head>
<title>PHP Form</title>
</head>
<body>
<form action="action.php" method="post">
 <p>Your name: <input type="text" name="name" /></p>
 <p>Your age: <input type="text" name="age" /></p>
 <p><input type="submit" /></p>
</form>
</body>
</html>
```

- ...and the PHP script called "action.php"

```php
Hi <?php echo htmlspecialchars($_POST['name']); ?>.
You are <?php echo (int)$_POST['age']; ?> years old.
```