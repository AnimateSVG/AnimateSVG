import os
from os import rename, remove
from shutil import copyfile
from datetime import datetime

from flask import Flask, flash, request, redirect, render_template, send_from_directory, url_for
from werkzeug.utils import secure_filename

from src.pipeline import *

UPLOAD_FOLDER = 'data/uploads'
ANIMATIONS_FOLDER = 'data/animations'
ALLOWED_EXTENSIONS = {'svg'}

app = Flask(__name__, static_folder='./static')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['ANIMATIONS_FOLDER'] = ANIMATIONS_FOLDER


def allowed_file(filename):
    """ Check if the uploaded file is a SVG

    Args:
        filename (str): Name of a file

    Returns:
        (boolean): True if the file is a SVG

    """
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)

        file = request.files['file']

        # Todo: Error handling
        # if file.filename == '':
            # flash('No selected file')
            # return redirect(request.url)

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            generator_animation_filename = filename.replace(".svg", "_animated_backprop.svg")
            optimizer_animation_filename = filename.replace(".svg", "_animated_entmoot.svg")

            logo = Logo(data_dir=os.path.join(app.config['UPLOAD_FOLDER'], filename))
            logo.animate()

            # copyfile(file_path, os.path.join(app.config['UPLOAD_FOLDER'], generator_animation_filename))
            # copyfile(file_path, os.path.join(app.config['UPLOAD_FOLDER'], optimizer_animation_filename))

            return redirect(url_for('show_animations', filename=filename))
    return render_template('index.html')


@app.route('/animations/<filename>', methods=['GET', 'POST'])
def show_animations(filename):
    generator_animation_filename = filename.replace(".svg", "_animated_backprop.svg")
    optimizer_animation_filename = filename.replace(".svg", "_animated_entmoot.svg")
    if request.method == 'POST':
        for model in ['optimizer', 'generator']:
            if request.form.get(model) in ['Very bad', 'Bad', 'Okay', 'Good', 'Very good']:
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                rating = request.form.get(model)
                with open('data/ratings.csv', 'a+') as f:
                    f.write(';'.join([timestamp, filename, model, rating]) + '\n')
    return render_template('show.html',
                           filename=filename,
                           animation_filename_genetic=generator_animation_filename,
                           animation_filename_entmoot=optimizer_animation_filename)


@app.route('/animate', methods=['GET', 'POST'])
def animate():
    if request.method == 'GET':
        # Todo: Error handling for GET request
        print('Not possible')
    return render_template('show.html')


@app.route('/data/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'],
                               filename)


@app.route('/data/uploads/<animation_filename_genetic>')
def animated_svg_genetic(animation_filename_genetic):
    return send_from_directory(app.config['UPLOAD_FOLDER'],
                               animation_filename_genetic)


@app.route('/data/uploads/<animation_filename_entmoot>')
def animated_svg_entmoot(animation_filename_entmoot):
    return send_from_directory(app.config['UPLOAD_FOLDER'],
                               animation_filename_entmoot)


@app.route('/redirect_button', methods=['GET'])
def redirect_button():
    return redirect(url_for('index', _anchor='upload'))


@app.route('/rating', methods=['GET', 'POST'])
def rating():
    print(request)
