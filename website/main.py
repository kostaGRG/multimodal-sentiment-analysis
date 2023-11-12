from flask import Flask,render_template,request, redirect, url_for, jsonify
import prediction
from PIL import Image
import io
import uuid
import os
from werkzeug.utils import secure_filename
from apscheduler.schedulers.background import BackgroundScheduler
import json

UPLOAD_FOLDER = 'static/uploads'
app = Flask(__name__, static_folder='static')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key='TbQuzX4xlOPv733TfK8abLlXVSFJvFOcQMnZ8WyMHBEB6aoVpCHzIfXrnqAE8Z4B'

BASE_URL = 'http://127.0.0.1:5001/'

def delete_files():
    folder_path = app.config['UPLOAD_FOLDER']
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
        except Exception as e:
            print(f"Error deleting {file_path}: {e}")

scheduler = BackgroundScheduler()
scheduler.start()
scheduler.add_job(delete_files, 'interval', days=1) 


##### WEBSITE #####
@app.route('/',methods=['POST','GET'])
def index():
    # try:
        if request.method == 'GET':
            return render_template('index.html')
        elif request.method == 'POST':
            text = request.form['text_input']
            image_name = request.files['image_input'].filename
            
            if text == '':
                text = None
            if image_name and prediction.allowed_file(image_name):
                filename = str(uuid.uuid4()) + secure_filename(image_name)
                image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                
                image = Image.open(io.BytesIO(request.files['image_input'].stream.read()))
                image.save(image_path)
                image_url = 'static/uploads/'+filename
            else:
                image = None
                image_url = None
            textPred, imagePred, finalPred, probs = prediction.makePredictions(text,image)
            probs = json.dumps(probs)
            
            return render_template('result.html', text=text, image=image_url, text_prediction=textPred, image_prediction = imagePred, final_prediction = finalPred,probs=probs)
    # except:
    #     print('Something went wrong in index.')
    

if __name__ == '__main__':
    app.run(host='localhost',port=5001,debug=True)