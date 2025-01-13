from flask import Flask, request, render_template, jsonify, send_from_directory
from werkzeug.utils import secure_filename
import os
from transformers import GPT2TokenizerFast, ViTImageProcessor, VisionEncoderDecoderModel, pipeline
from PIL import Image
import torch

# Load the ViT-GPT2 model and tokenizer
vit_model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
vit_tokenizer = GPT2TokenizerFast.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
vit_image_processor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

# Load the GIT-large-COCO model
caption_generator_git = pipeline("image-to-text", model="microsoft/git-large-coco")

# Initialize Flask application
app = Flask(__name__)

DATASET_FOLDER = "dataset"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}

app.config['DATASET_FOLDER'] = DATASET_FOLDER

if not os.path.exists(DATASET_FOLDER):
    os.makedirs(DATASET_FOLDER)


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def load_dataset():
    images = [f for f in os.listdir(DATASET_FOLDER) if allowed_file(f)]
    return images


@app.route("/", methods=["GET", "POST"])
def upload_or_choose_image():
    if request.method == "POST":
        if "dataset_image" in request.form:
            dataset_image = request.form["dataset_image"]
            file_path = os.path.join(app.config['DATASET_FOLDER'], dataset_image)
            if os.path.exists(file_path):
                return generate_captions({"image_path": file_path})
            else:
                return render_template("index.html", error="File not found in dataset", dataset_images=load_dataset())

        return render_template("index.html", error="Invalid selection", dataset_images=load_dataset())

    dataset_images = load_dataset()
    return render_template("index.html", dataset_images=dataset_images)


@app.route("/generate_captions", methods=["POST"])
def generate_captions(data=None):
    if not data:
        data = request.json

    image_path = data.get("image_path")
    if not image_path or not os.path.exists(image_path):
        return jsonify({"status": "error", "message": "Invalid image path"})

    try:
        image = Image.open(image_path)

        # Generate caption using ViT-GPT2 model
        pixel_values = vit_image_processor(image, return_tensors="pt").pixel_values
        generated_ids = vit_model.generate(pixel_values)
        vit_caption = vit_tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

        # Generate caption using GIT-large-COCO model
        git_caption = caption_generator_git(image)[0]['generated_text']

        captions = {
            "ViT-GPT2": vit_caption,
            "GIT-large-COCO": git_caption,
            "Model_3": "Placeholder caption from Model 3",
            "Model_4": "Placeholder caption from Model 4",
        }

        return render_template("result.html", image_path=image_path, captions=captions)

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})


@app.route("/dataset/images")
def dataset_images_popup():
    dataset_images = load_dataset()
    return render_template("popup.html", dataset_images=dataset_images)


@app.route("/dataset/<filename>")
def dataset_file(filename):
    return send_from_directory(DATASET_FOLDER, filename)


if __name__ == "__main__":
    app.run(debug=True)
