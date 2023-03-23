import os
import openai
from flask import Flask, redirect, render_template, request, url_for

# Create Flask app
app = Flask(__name__)

# Indicate OpenAI API Key environmental variable
openai.api_key = os.getenv("OPENAI_API_KEY")

# Specify root route
@app.route("/", methods=("GET", "POST"))
def index():
    if request.method == "POST":
        topic = request.form["topic"]
        response = openai.Completion.create(
            model="text-davinci-003",
            prompt=generate_prompt(topic),
            temperature=0.6,
        )
        return redirect(url_for("index", result=response.choices[0].text))

    result = request.args.get("result")
    return render_template("index.html", result=result)


def generate_prompt(animal):
    return """Generate a quiz question about the {}""".format(topic)

#FIN