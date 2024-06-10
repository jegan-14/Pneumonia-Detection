from flask import Flask, request, jsonify
import utils

app = Flask(__name__)


@app.route("/", methods=["GET"])
def sample():
    return "hi"


@app.route("/classify_image", methods=["GET", "POST"])
def classify_image():
    img_data = request.form["image_data"]
    response = jsonify(utils.classify_image(img_data))
    response.headers.add("Access-Control-Allow-Origin", "*")
    return response


if __name__ == "__main__":
    app.run(port=5000)
