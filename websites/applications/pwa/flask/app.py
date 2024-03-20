from flask import Flask, render_template, send_file

app = Flask(__name__)

# Serve manifest (for PWA)
@app.route('/manifest.json')
def serve_manifest():
    return send_file('manifest.json', mimetype='application/manifest+json')

# Serve service orker
@app.route('/service_worker.js')
def serve_sw():
    return send_file('sw.js', mimetype='application/javascript')

# Serve home
@app.route('/')
def index():
    return render_template('index.html')
if __name__ == '__main__':
    app.run()



#FIN