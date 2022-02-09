import numpy as np
from flask import Flask, render_template, request
from keras.models import load_model
from keras.preprocessing import image

app = Flask(__name__)

dic = {
  0: 'Normal',
  1: 'Pneumonia',
}

model = load_model('Pneumonia_detector.h5')

model.make_predict_function()

def predict_label(img_path):
  i = image.load_img(img_path, target_size=(64,64))
  i = image.img_to_array(i)/255.0
  i = i.reshape(1, 64, 64,3)
  p = model.predict(i)
  return dic[np.round(p[0][0])]

print('Normal:', predict_label('../img-1.jpeg'))
print('Pneumonia:', predict_label('../img-2.jpeg'))

# routes
@app.route("/", methods=['GET', 'POST'])
def main():
	return render_template("index.html")

@app.route("/about")
def about_page():
	return "Please subscribe  Artificial Intelligence Hub..!!!"

@app.route("/submit", methods = ['GET', 'POST'])
def get_output():
	if request.method == 'POST':
		img = request.files['my_image']

		img_path = "static/" + img.filename	
		img.save(img_path)

		p = predict_label(img_path)

	return render_template("index.html", prediction = p, img_path = img_path)


if __name__ =='__main__':
	#app.debug = True
	app.run(debug = True)