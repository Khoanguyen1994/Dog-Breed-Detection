import os
from flask import Flask, render_template, request
from dog_app import dog_human_detector

app = Flask(__name__)

UPLOAD_FOLDER = os.path.join(os.getcwd(), 'upload_folder')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['ALLOWED_EXTENSIONS'] = ALLOWED_EXTENSIONS

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    try:
        file = request.files['file']
        if file and allowed_file(file.filename):
            filename = file.filename
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            img_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            breed = dog_human_detector(img_path)
            return render_template('result.html', breed=breed)
        else:
            return render_template('index.html', error='Invalid file type. Please use a file with one of the following extensions: png, jpg, jpeg, gif')
    except Exception as e:
        print(e)
        return render_template('index.html', error='An error occurred. Please try a different file.')

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8080, debug=False, threaded=False)
