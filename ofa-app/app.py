from flask import Flask, request, jsonify
from PIL import Image
from torchvision import transforms
from transformers import OFATokenizer, OFAModel
import os

app = Flask(__name__)

# Load ofa-large-caption model
ckpt_dir = "./OFA-large-caption"
mean, std = [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]
resolution = 480
patch_resize_transform = transforms.Compose([
    lambda image: image.convert("RGB"),
    transforms.Resize((resolution, resolution), interpolation=Image.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std)
])
tokenizer = OFATokenizer.from_pretrained(ckpt_dir)
model = OFAModel.from_pretrained(ckpt_dir)
model.eval() 

# endpoint for caption generating 
@app.route("/generate_caption", methods=["POST"])
def generate_caption():
    data = request.json
    image_path = data.get("image_path")
    image_path = os.path.join("..", image_path)

    if not image_path or not os.path.exists(image_path):
        return jsonify({"error": "Invalid image path"}), 400

    try:
        # Process image and generate caption
        img = Image.open(image_path)
        patch_img = patch_resize_transform(img).unsqueeze(0)

        prompt = " what does the image describe?"
        inputs = tokenizer([prompt], return_tensors="pt").input_ids
        outputs = model.generate(
            inputs,
            patch_images=patch_img,
            num_beams=5,
            no_repeat_ngram_size=3,
            max_length=16
        )

        caption = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return jsonify({"caption": caption})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=False)
