# from flask import Flask, render_template, request, send_file
# import os
# import numpy as np
# import matplotlib.pyplot as plt
# from PIL import Image
# from segment import U_Net_Generator, preprocess_image

# app = Flask(__name__)

# # Set the path to the uploaded images and the output directory
# UPLOAD_FOLDER = 'uploads'
# OUTPUT_FOLDER = 'results'
# app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
# app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER

# # Ensure the upload and output folders exist
# os.makedirs(UPLOAD_FOLDER, exist_ok=True)
# os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# # Load the U-Net model
# img_size = 256
# image_shape = (img_size, img_size, 1)  # Adjust the input shape according to your dataset
# model = U_Net_Generator(image_shape)
# model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
# model.load_weights('model_nerveAdam2.h5')


# def process_image(input_path):
#     processed_image = preprocess_image(input_path)
#     predicted_mask = model.predict(processed_image)

#     img_arr = np.array(processed_image)[0]
#     image_mask = np.array(predicted_mask)[0]

#     return img_arr, image_mask


# @app.route('/')
# def home():
#     return render_template('index.html')


# @app.route('/upload', methods=['POST'])
# def upload_file():
#     if 'file' not in request.files:
#         return render_template('index.html', error='No file part')

#     file = request.files['file']

#     if file.filename == '':
#         return render_template('index.html', error='No selected file')

#     if file:
#         file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
#         file.save(file_path)

#         img_arr, image_mask = process_image(file_path)

#         output_path = os.path.join(app.config['OUTPUT_FOLDER'], 'result.png')

#         fig, ax = plt.subplots(1, 3, figsize=(16, 12))
#         ax[0].imshow(img_arr, cmap='gray')
#         ax[0].set_title('Original')

#         ax[1].imshow(image_mask, cmap='gray')
#         ax[1].set_title('Mask')

#         ax[2].imshow(img_arr, cmap='gray', interpolation='none')
#         ax[2].imshow(image_mask, interpolation='none', alpha=0.7)
#         ax[2].set_title('Mask overlay')

#         plt.savefig(output_path)

#         return send_file(output_path, mimetype='image/png')


# if __name__ == '__main__':
#     app.run(debug=True)






from flask import Flask, render_template, request, send_file
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from segment import U_Net_Generator, preprocess_image

app = Flask(__name__)

# Set the path to the uploaded images and the output directory
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'results'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER

# Ensure the upload and output folders exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Load the U-Net model
img_size = 256
image_shape = (img_size, img_size, 1)  # Adjust the input shape according to your dataset
model = U_Net_Generator(image_shape)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.load_weights('model_nerveAdam2.h5')


def process_image(input_path, output_folder):
    processed_image = preprocess_image(input_path)
    predicted_mask = model.predict(processed_image)

    img_arr = np.array(processed_image)[0]
    image_mask = np.array(predicted_mask)[0]

    output_path = os.path.join(output_folder, 'result.png')
    print("Output path: ", output_path)

    fig, ax = plt.subplots(1, 3, figsize=(16, 12))
    ax[0].imshow(img_arr, cmap='gray')
    ax[0].set_title('Original')

    ax[1].imshow(image_mask, cmap='gray')
    ax[1].set_title('Mask')

    ax[2].imshow(img_arr, cmap='gray', interpolation='none')
    ax[2].imshow(image_mask, interpolation='none', alpha=0.7)
    ax[2].set_title('Mask overlay')

    plt.savefig(output_path)

    return output_path


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return render_template('index.html', error='No file part')

    file = request.files['file']

    if file.filename == '':
        return render_template('index.html', error='No selected file')

    if file:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)

        result_path = process_image(file_path, app.config['OUTPUT_FOLDER'])
        # result_download_path = '/download/' + os.path.basename(result_path)
        result_filename = os.path.basename(result_path)
        return render_template('index.html', result_path=result_path, result_filename=result_filename)


@app.route('/download/<filename>')
def download_file(filename):
    path = os.path.join(app.config['OUTPUT_FOLDER'], filename)
    return send_file(path, as_attachment=True)


if __name__ == '__main__':
    app.run(debug=True)

