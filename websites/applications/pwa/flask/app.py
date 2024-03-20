import os
from flask import Flask, render_template, send_file
from flask import Flask, flash, request, redirect, url_for
from werkzeug.utils import secure_filename

# Define constants
UPLOAD_FOLDER = '/home/kampff/NoBlackBoxes/OtherBlackBoxes/websites/applications/pwa/flask/_tmp'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Serve manifest (for PWA)
@app.route('/manifest.json')
def serve_manifest():
    return send_file('manifest.json', mimetype='application/manifest+json')

# Serve service orker
@app.route('/service_worker.js')
def serve_sw():
    return send_file('sw.js', mimetype='application/javascript')

# Serve home
@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            return render_template('index.html')
    return render_template('index.html')

if __name__ == '__main__':
    app.run()



#FIN