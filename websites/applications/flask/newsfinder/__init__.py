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
root_path = "/var/www/llm/newsfinder"
openai.api_key_path = root_path + "/.key"

# State
model = "text-curie-001"
generation = "GPT-3"
result = ""

# Specify root route
@app.route("/", methods=("GET", "POST"))
def index():
    global result
    if request.method == "POST":
        topic = request.form["topic"]
        response = openai.Completion.create(
            model="text-davinci-003",
            prompt=generate_prompt(topic),
            temperature=0.6,
            max_tokens=256,
        )
        result =  response.choices[0].text

    print(result)
    return render_template("index.html", result=result)

def generate_prompt(topic):
#    return """Generate a few paragraphs describing {}, in the style of a wikipedia introductory section, but written for a 5 year old""".format(topic)
    return """Generate an article talking about {}, make it realistic and believable sign at the end with the name of a popular journalist. make the story as if it was 2023""".format(topic)

# MAIN
if __name__ == "__main__":
    app.run()

#FIN