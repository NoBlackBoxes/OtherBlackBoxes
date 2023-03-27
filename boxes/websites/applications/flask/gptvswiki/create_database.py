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
