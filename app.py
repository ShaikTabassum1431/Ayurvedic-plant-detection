from flask import Flask, request, render_template, redirect, url_for
import os
import tensorflow as tf
import numpy as np

app = Flask(__name__,static_folder='static')
# Ensure the 'uploads' directory exists
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
# Route to render the upload form
@app.route('/')
def upload_form():
    return render_template('index1.html')
@app.route('/1stpage',methods=['GET','POST'])
def open1():
    return render_template('index.html')
@app.route('/2ndpage',methods=['GET','POST'])
def open2():
    return render_template('index2.html')

#Route to handle question
@app.route('/answer', methods=['POST'])
def answer_question():
    if request.method == 'POST':
        # Get the value from the form with the name 'qa'
        text_input = str(request.form.get('qa'))

        text_input=text_input.lower()
        if text_input=="shatavari":
            return render_template('shatavari.html',answer=url_for('static',filename="shatavari.jpeg"))
        elif text_input=="turmeric":
            return render_template('turmeric.html',answer=url_for('static',filename="turmeric.jpeg"))
        elif text_input=="manjistha":
            return render_template('manjistha.html',answer=url_for('static',filename="manjistha.jpeg"))
        elif text_input=="brahmi":
            return render_template('brahmi.html',answer=url_for('static',filename="brahmi.jpeg"))
        elif text_input=="ashwadgama":
            return render_template('ashwadgama.html',answer=url_for('static',filename="ashwadgama.jpeg"))
        elif text_input=="aloevara":
            return render_template('alovera.html',answer=url_for('static',filename="aloevara.jpeg"))
        elif text_input=="tulsi":
            return  render_template('tulsi.html',answer=url_for('static',filename="tulsi.jpg"))
        elif text_input=="neem":
            return render_template('neem.html',answer=url_for('static',filename="neem.jpg"))
        elif text_input=="nagfani":
            return render_template('nagfani.html',answer=url_for('static',filename="nagfani.jpg"))
        

#Route to handle file upload
@app.route('/index', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return 'No file part'
    
    file = request.files['file']
    
    if file.filename == '':
        return 'No selected file'
    
    if file:
        # Save file to the uploads directory
        saved_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(saved_path)
        
        # Print the file path on the server where the file is saved
        print(f"File saved to: {saved_path}")

        # Define the image dimensions and paths
        img_width = 180
        img_height = 180
        data_cat = ['Manjistha', 'aloevara', 'ashwadgama', 'brahmi', 'nagfani', 'neem', 'shatavari', 'tulsi', 'turmeric']  # Replace with actual categories

        # Load the saved model
        model = tf.keras.models.load_model("model path")

        # Predict the image
        image_path = saved_path
        image = tf.keras.utils.load_img(image_path, target_size=(img_width, img_height))

        # Convert image to array format and prepare it for prediction
        img_arr = tf.keras.utils.img_to_array(image)
        img_bat = tf.expand_dims(img_arr, 0)  # Add batch dimension

        # Predict using the loaded model
        predict = model.predict(img_bat)
        score = tf.nn.softmax(predict)

        # Create the output string with the category name and accuracy
        s = 'Plant in image is {} with accuracy of {:0.2f}%'.format(data_cat[np.argmax(score)], np.max(score) * 100)
        l=["Manjistha is renowned for its blood-purifying properties in Ayurvedic medicine. It is often used to treat skin conditions, improve complexion, and support healthy lymphatic drainage. Additionally, it helps with disorders like eczema, acne, and pigmentation issues.",
           " Aloe vera is widely used for its healing properties, particularly for skin care. It soothes burns, reduces inflammation, and promotes wound healing. Aloe vera is also consumed to aid digestion and detoxify the body.",
           "Ashwagandha is a powerful adaptogen in Ayurveda, helping the body manage stress and anxiety. It improves energy levels, boosts the immune system, and enhances memory and cognitive function. It is also used to support reproductive health.",
           "Brahmi is valued for its ability to enhance cognitive function, improve memory, and reduce stress. It is also used in treating mental disorders such as anxiety and depression. Brahmi is known to support overall brain health and improve concentration.",
           "Nagfani is used to treat inflammation, reduce cholesterol, and manage diabetes. The cactus is also known for its antioxidant properties and is often used to aid digestion and detoxify the body. It is effective in soothing skin irritations and burns.",
           "Neem is a versatile medicinal plant with antibacterial, antifungal, and antiviral properties. It is used to treat skin disorders, purify the blood, and boost the immune system. Neem oil is widely used in skincare for acne, eczema, and other conditions. Neem is also used in oral care products like toothpaste.",
           "Shatavari is known as a rejuvenating tonic for women’s health. It supports reproductive health, balances hormones, and promotes lactation in nursing mothers. It is also used to enhance vitality and boost immunity. Shatavari is known for its ability to balance the body and improve energy levels.",
           " Tulsi is a sacred herb in Hinduism and is known for its adaptogenic properties. It helps the body adapt to stress and has a calming effect on the mind. Tulsi is also used to treat respiratory issues, support digestion, and strengthen the immune system. It is considered a powerful detoxifier and is often used in teas and herbal remedies.",
           "Turmeric is widely known for its anti-inflammatory and antioxidant properties, largely due to its active compound, curcumin. It is used to treat a wide range of ailments, including joint pain, digestive issues, and skin problems. Turmeric is also known to boost immunity, support liver function, and improve cardiovascular health. It is commonly used in cooking as well as in medicinal formulations."]
        if data_cat[np.argmax(score)]=="Manjistha":
           return render_template('result.html', result=s,mat=l[0])
        elif data_cat[np.argmax(score)]=="aloevara":
           return render_template('result.html', result=s,mat=l[1])
        elif data_cat[np.argmax(score)]=="ashwadgama":
           return render_template('result.html', result=s,mat=l[2])
        elif data_cat[np.argmax(score)]=="brahmi":
           return render_template('result.html', result=s,mat=l[3])
        elif data_cat[np.argmax(score)]=="nagfani":
           return render_template('result.html', result=s,mat=l[4])
        elif data_cat[np.argmax(score)]=="neem":
           return render_template('result.html', result=s,mat=l[5])
        elif data_cat[np.argmax(score)]=="shatavari":
           return render_template('result.html', result=s,mat=l[6])
        elif data_cat[np.argmax(score)]=="tulsi":
           return render_template('result.html', result=s,mat=l[7])
        elif data_cat[np.argmax(score)]=="turmeric":
           return render_template('result.html', result=s,mat=l[8])


if __name__ == "__main__":
    app.run(debug=True)
