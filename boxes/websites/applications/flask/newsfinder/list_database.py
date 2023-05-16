import sqlite3

# Connect to database
conn = sqlite3.connect('_tmp/database.db')

# Some kind of SQL cursor
conn.row_factory = sqlite3.Row

# Extract data about answers from database
answers = conn.execute('SELECT * FROM answers').fetchall()

# Display
for answer in answers:
    print("{0} ({1}) : {2}\n Wiki: {3}\n ---\n GPT: {4}\n\n".format(answer['id'], answer['correct'], answer['model'], answer['wiki'][:16], answer['gpt'][:16]))

# Close connection
conn.close()