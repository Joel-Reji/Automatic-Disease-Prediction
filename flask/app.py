from flask import Flask, flash, request, redirect, url_for, render_template, logging
import urllib.request
import os
from werkzeug.utils import secure_filename
import cv2
import pickle
import imutils
import sklearn
from tensorflow.keras.models import load_model
# from pushbullet import PushBullet
import joblib
import numpy as np
import speech_recognition as sr
from tensorflow.keras.applications.vgg16 import preprocess_input


covid_model = load_model('models/covid.h5')
heart_model = pickle.load(open('models/heart_disease.pickle.dat', "rb"))
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])

app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = "secret key"

############################################# BRAIN TUMOR FUNCTIONS ################################################


def preprocess_imgs(set_name, img_size):
    """
    Resize and apply VGG-15 preprocessing
    """
    set_new = []
    for img in set_name:
        img = cv2.resize(img,dsize=img_size,interpolation=cv2.INTER_CUBIC)
        set_new.append(preprocess_input(img))
    return np.array(set_new)


def crop_imgs(set_name, add_pixels_value=0):
    """
    Finds the extreme points on the image and crops the rectangular out of them
    """
    set_new = []
    for img in set_name:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        # threshold the image, then perform a series of erosions +
        # dilations to remove any small regions of noise
        thresh = cv2.threshold(gray, 45, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.erode(thresh, None, iterations=2)
        thresh = cv2.dilate(thresh, None, iterations=2)

        # find contours in thresholded image, then grab the largest one
        cnts = cv2.findContours(
            thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        c = max(cnts, key=cv2.contourArea)

        # find the extreme points
        extLeft = tuple(c[c[:, :, 0].argmin()][0])
        extRight = tuple(c[c[:, :, 0].argmax()][0])
        extTop = tuple(c[c[:, :, 1].argmin()][0])
        extBot = tuple(c[c[:, :, 1].argmax()][0])

        ADD_PIXELS = add_pixels_value
        new_img = img[extTop[1]-ADD_PIXELS:extBot[1]+ADD_PIXELS,
                      extLeft[0]-ADD_PIXELS:extRight[0]+ADD_PIXELS].copy()
        set_new.append(new_img)

    return np.array(set_new)

##################################################################################################


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS


@app.route('/')
def home():
    return render_template('homepage.html')


@app.route('/covid')
def covid():
    return render_template('covid.html')
@app.route('/heartdisease')
def heartdisease():
    return render_template('heartdisease.html')


@app.route('/audio_to_text/')
def audio_to_text():
    flash(" Press Start to start recording audio and press Stop to end recording audio")
    return render_template('audio_to_text.html')

###################################################################


@app.route('/audio', methods=['POST'])
def audio():
    r = sr.Recognizer()
    label=['no smell', 'no taste']
    with open('upload/audio.wav', 'wb') as f:
        f.write(request.data)
  
    with sr.AudioFile('upload/audio.wav') as source:
        audio_data = r.record(source)
        text = r.recognize_google(audio_data, language='en-IN')
        print(text)
        
        return_text = " your symptoms are : <br> "
        try:
            if text in label:
                re='Result: Covid Positive <br>'
                return_text =return_text+ text+ " <br> "+re
            else:
                re='Result: Covid Negative <br>'
                return_text =return_text+ text+ " <br> "+re
                
        except:
            return_text = " Sorry!!!! Voice not Detected "
        
    return str(return_text)


@app.route('/resultc', methods=['POST'])
def resultc():
    if request.method == 'POST':
        firstname = request.form['firstname']
        lastname = request.form['lastname']
        email = request.form['email']
        phone = request.form['phone']
        gender = request.form['gender']
        age = request.form['age']
        file = request.files['file']
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            flash('Image successfully uploaded and displayed below')
            img = cv2.imread('static/uploads/'+filename)
            img = cv2.resize(img, (224, 224))
            img = img.reshape(1, 224, 224, 3)
            img = img/255.0
            pred = covid_model.predict(img)
            if pred < 0.5:
                pred = 0
            else:
                pred = 1
            # pb.push_sms(pb.devices[0],str(phone), 'Hello {},\nYour COVID-19 test results are ready.\nRESULT: {}'.format(firstname,['POSITIVE','NEGATIVE'][pred]))
            return render_template('resultc.html', filename=filename, fn=firstname, ln=lastname, age=age, r=pred, gender=gender)

        else:
            flash('Allowed image types are - png, jpg, jpeg')
            return redirect(request.url)


@app.route('/resulth', methods=['POST'])
def resulth():
    if request.method == 'POST':
        firstname = request.form['firstname']
        lastname = request.form['lastname']
        email = request.form['email']
        phone = request.form['phone']
        gender = request.form['gender']
        nmv = float(request.form['nmv'])
        tcp = float(request.form['tcp'])
        eia = float(request.form['eia'])
        thal = float(request.form['thal'])
        op = float(request.form['op'])
        mhra = float(request.form['mhra'])
        age = float(request.form['age'])
        print(np.array([nmv, tcp, eia, thal, op, mhra, age]).reshape(1, -1))
        pred = heart_model.predict(
            np.array([nmv, tcp, eia, thal, op, mhra, age]).reshape(1, -1))
        # pb.push_sms(pb.devices[0],str(phone), 'Hello {},\nYour Diabetes test results are ready.\nRESULT: {}'.format(firstname,['NEGATIVE','POSITIVE'][pred]))
        return render_template('resulth.html', fn=firstname, ln=lastname, age=age, r=pred, gender=gender)


# No caching at all for API endpoints.
@app.after_request
def add_header(response):
    """
    Add headers to both force latest IE rendering engine or Chrome Frame,
    and also to cache the rendered page for 10 minutes.
    """
    response.headers['X-UA-Compatible'] = 'IE=Edge,chrome=1'
    response.headers['Cache-Control'] = 'public, max-age=0'
    return response


if __name__ == '__main__':
    app.run(debug=True)
