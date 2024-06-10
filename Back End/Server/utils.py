import torch
from torchvision import transforms
from PIL import Image
import base64
import io
from Model import Model

transform = transforms.Compose(
    [transforms.Resize((128, 128)), transforms.Grayscale(), transforms.ToTensor()]
)

class_labels = {0: "NORMAL", 1: "PNEUMONIA"}


def convert_b64_to_PIL(base64_str):
    if base64_str.startswith("data:image"):
        base64_str = base64_str.split(",")[1]
    bytes = base64.b64decode(base64_str)
    byte_stream = io.BytesIO(bytes)
    img = Image.open(byte_stream)
    return img


def get_base64_test():
    with open("Back End/Server/base64.txt") as f:
        return f.read()


def preprocess(img):
    preprocessed_img = transform(img)
    return preprocessed_img


def get_predictions(img):
    model = Model(1, 1, 10)
    model.load_state_dict(torch.load("../model.pth", map_location=torch.device("cpu")))
    model.eval()
    with torch.inference_mode():
        logits = model(img)
        preds = torch.sigmoid(logits)
    return preds


def classify_image(b64_path, path=None):
    if path:
        img = Image.open(path)
    else:
        img = convert_b64_to_PIL(b64_path)
    preprocessed_img = preprocess(img)
    batchified_img = preprocessed_img.unsqueeze(1)
    preds = get_predictions(batchified_img)
    cls_idx = int(torch.round(preds).item())
    return {"class": class_labels[cls_idx], "pred_prob": preds.item()}


if __name__ == "__main__":
    print(classify_image(get_base64_test()))
