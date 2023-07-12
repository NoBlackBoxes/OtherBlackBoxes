import os
from flask import Flask, redirect, render_template, request, url_for

# Create Flask app
app = Flask(__name__)

# Specify root route
@app.route("/", methods=("GET", "POST"))
def index():
    if request.method == "POST":
        direction = request.form["direction"]
        print(direction)

    return render_template("index.html")

#FIN