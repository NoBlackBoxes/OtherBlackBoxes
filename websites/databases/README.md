# boxes : websites : databases

A database can either be relational (assumes a table structure) or non-relational (anything goes).

## SQLite
Implements the SQL (query language), minmally and locally. It is included with Python3.

Create a schema (structure) file called "schema.sql" with the following contents:

```sql
DROP TABLE IF EXISTS users;        --Delete pre-existing "user" table

CREATE TABLE users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    name TEXT NOT NULL,
    email TEXT NOT NULL,
    age INTEGER NOT NULL
);
```

Write a python script called "init_database.py" to initialize the database:

```python
import sqlite3

# Connect to database (it will create the file on first connection)
connection = sqlite3.connect('database.db')

# Execute the SQL commands in the schema file
with open('schema.sql') as f:
    connection.executescript(f.read())

# Commit changes
connection.commit()

# Close connection
connection.close()
```

Add some data (users) to your database:

```python
import sqlite3
import argparse

# Parse command line
parser = argparse.ArgumentParser(description='Add user to database.')
parser.add_argument('name', metavar='Name', type=str, nargs=1, help='full name of new user')
parser.add_argument('email', metavar='Email', type=str, nargs=1, help='email address of new user')
parser.add_argument('age', metavar='Age', type=int, nargs=1, help='age of user')
args = parser.parse_args()
name = args.name[0]
email = args.email[0]
age = args.age[0]

# Connect to database (it will create the file on first connection)
connection = sqlite3.connect('database.db')

# Some kind of SQL cursor
cur = connection.cursor()

# Insert data about user from command line arguments
cur.execute("INSERT INTO users (name, email, age) VALUES (?, ?, ?)", (name, email, age))

# Commit changes
connection.commit()

# Close connection
connection.close()
```

Now list all the users:

```python
import sqlite3

# Connect to database (it will create the file on first connection)
conn = sqlite3.connect('database.db')

# Some kind of SQL cursor
conn.row_factory = sqlite3.Row

# Extract data about users from database
users = conn.execute('SELECT * FROM users').fetchall()

# Display
for user in users:
    print("{0} ({1}) : {2} ".format(user['name'], user['email'], user['age']))

# Close connection
conn.close()
```