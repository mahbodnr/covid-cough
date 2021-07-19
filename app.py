from flask import Flask, render_template, url_for, request, redirect
from flask_bootstrap import Bootstrap

import os
import json
import main

app = Flask(__name__)
Bootstrap(app)

"""
Routes
"""
@app.route('/', methods=['GET','POST'])
def index():
    if request.method == 'POST':
        try:
            uploaded_file = request.files['file']
            if uploaded_file.filename != '':
                audio_path = os.path.join('static', uploaded_file.filename)
                # uploaded_file.save(audio_path)
                class_label, class_name = main.run(audio_path)
                result = json.dumps({
                    'class_label': class_label,
                    'class_name': class_name,
                })
                return result
                # return render_template('result.html', result = result)
        except Exception as e:
            import traceback
            return traceback.format_exc()
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug = False)