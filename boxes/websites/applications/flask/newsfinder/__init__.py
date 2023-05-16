import os
import time
import random
import openai
import sqlite3
from flask import Flask, redirect, render_template, request, url_for
import utilities as utl

# Seed random
random.seed(time.time())

# Create Flask app
app = Flask(__name__)

# Indicate OpenAI API Key environmental variable
#openai.api_key = os.getenv("OPENAI_API_KEY")
root_path = "/var/www/llm/gptvswiki"
openai.api_key_path = root_path + "/.key"

# State
is_wiki_1 = True
num_correct = 0
num_wrong = 0
model = "text-curie-001"
generation = "GPT-3"
gpt_text = ''
wiki_text = ''

# Specify root route
@app.route("/", methods=("GET", "POST"))
def index():
    global num_correct, num_wrong

    # Reset score counter
    num_correct = 0
    num_wrong = 0

    return redirect(url_for("select"))

# Specify select routes
@app.route("/select1", methods=("GET", "POST"))
def select1():
    global is_wiki_1, num_correct, num_wrong
    if is_wiki_1:
        num_correct += 1
        return redirect(url_for("correct"))
    else:
        num_wrong += 1
        return redirect(url_for("wrong"))

@app.route("/select2", methods=("GET", "POST"))
def select2():
    global is_wiki_1, num_correct, num_wrong
    if is_wiki_1:
        num_wrong += 1
        return redirect(url_for("wrong"))
    else:
        num_correct += 1
        return redirect(url_for("correct"))

@app.route("/wrong", methods=("GET", "POST"))
def wrong():
    # Store wrong answer in database: timestamp, wiki, gpt, correct (False)
    global model, wiki_text, gpt_text
    database = sqlite3.connect(root_path + '/_tmp/database.db')
    cursor = database.cursor()
    cursor.execute("INSERT INTO answers (model, wiki, gpt, correct) VALUES (?, ?, ?,?)", (model, wiki_text, gpt_text, 0))
    database.commit()
    database.close()
    return render_template("wrong.html"), {"Refresh": "0.25; url=select"}

@app.route("/correct", methods=("GET", "POST"))
def correct():
    # Store correct answer in database: timestamp, wiki, gpt, corect (True)
    global model, wiki_text, gpt_text
    database = sqlite3.connect(root_path + '/_tmp/database.db')
    cursor = database.cursor()
    cursor.execute("INSERT INTO answers (model, wiki, gpt, correct) VALUES (?, ?, ?,?)", (model, wiki_text, gpt_text, 1))
    database.commit()
    database.close()
    return render_template("correct.html"), {"Refresh": "0.25; url=select"}

@app.route("/select", methods=("GET", "POST"))
def select():
    global is_wiki_1, num_correct, num_wrong, model, generation, gpt_text, wiki_text

    # If POST, update model/generation
    if request.method == "POST":
        generation = request.form.get('version')
        if generation == "GPT-4":
            model = "gpt-4"
        elif generation == "GPT-3.5":
            model = "text-davinci-003"        
        else:
            generation = "GPT-3"
            model = "text-curie-001"

    # Retrieve a valid Wikipedia extract
    while True:
        title = utl.random_wiki_title()
        extract = utl.request_wiki_extract(title)
        sentences  = utl.sanitize_text(extract)
        if utl.is_valid_text(sentences):
            break

    # Truncate Wikipedia extract
    wiki_text = utl.truncate_text(sentences)

    # Generate GPT prompt
    prompt = utl.generate_gpt_prompt(title)

    # Request GPT completetion
    gpt_text = utl.request_gpt_completion(prompt, model)

    # Sanitize and truncate GPT extract
    gpt_text  = utl.sanitize_text(gpt_text)
    gpt_text = utl.truncate_text(gpt_text)

    # Report
    print("\n\Title: {0}\n----\nGPT: {1}\n-----\nWiki: {2}\n\n{3} vs {4}".format(title, gpt_text, wiki_text, num_correct, num_wrong))

    # Randomize position
    if bool(random.getrandbits(1)):
        result1 = gpt_text
        result2 = wiki_text
        is_wiki_1 = False
    else:
        result1 = wiki_text
        result2 = gpt_text
        is_wiki_1 = True

    return render_template("index.html", title=title, generation=generation, result1=result1, result2=result2, num_correct=num_correct, num_trials=(num_correct+num_wrong))

# MAIN
if __name__ == "__main__":
    app.run()

#FIN