# Import Statements
from flask import Flask, render_template,request
from ultralytics import YOLO
from transformers import ViTImageProcessor, ViTForImageClassification # ViTFeatureExtractor
from PIL import Image
import warnings

warnings.filterwarnings('ignore')

app = Flask(__name__)

uploaded_image = None

@app.route("/")
def home():
    global uploaded_image
    uploaded_image = None
    return render_template('home.html')

@app.route("/upload_image", methods=['POST'])
def upload_image():
    global uploaded_image
    
    if request.method == 'POST' and 'imageUpload' in request.files:
        image = request.files['imageUpload']
        image.save('static/images/uploaded_image.jpg')
        uploaded_image = '../static/images/uploaded_image.jpg'
    return render_template('home.html', uploaded_image=uploaded_image)

@app.route("/model", methods=['POST'])
def model():
    global uploaded_image

    if request.method == 'POST' and uploaded_image != None:
        selected_model = request.form['model_type']
        uploaded_image = 'static/images/uploaded_image.jpg'
        
        if selected_model == 'Yolo Model':
            # load in pretrained YOLOv8n model
            yolo = YOLO('yolov8n.pt')

            # run inference on uploaded image
            results = yolo(uploaded_image,verbose=False) # results list
            
            # save results as image
            for r in results:
                img_array = r.plot()  # plot a BGR numpy array of predictions
                img = Image.fromarray(img_array[..., ::-1])  # RGB PIL image
                img.save('static/images/results.png')  # save image
                
            prediction='../static/images/results.png'
            return render_template('prediction.html', model=selected_model, yolo_prediction=prediction)

        elif selected_model == 'Vision Transformer Model':
            img = Image.open(uploaded_image)
            
            # instantiate the feature extractor specific to the model checkpoint
            feature_extractor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')

            # instantiate the pretrained model
            vit = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')

            # extract features (patches) from the image
            inputs = feature_extractor(images=img, return_tensors="pt")

            # predict by feeding the model (** is a python operator which unpacks the inputs)
            outputs = vit(**inputs)

            # convert outputs to logis
            logits = outputs.logits

            # model predicts one of the classes by pick the logit which has the highest probability
            predicted_class_idx = logits.argmax(-1).item()
            
            prediction = vit.config.id2label[predicted_class_idx]
            return render_template('prediction.html', model=selected_model, vit_prediction=prediction)
    else: 
        message = "Please upload an image before making prediction." 
        return render_template('prediction.html', model=message)

app.run(debug=False,port=5000)