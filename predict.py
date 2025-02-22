import argparse
import model_utils
import json

parser = argparse.ArgumentParser(description='Predicting flower name from an image along with the probability of that name.')
parser.add_argument('image_path')
parser.add_argument('checkpoint', default = 'checkpoint.pth')
parser.add_argument('--top_k', default = 5)
parser.add_argument('--category_names', default = 'cat_to_name.json')

args = parser.parse_args()

image_path = args.image_path
checkpoint = args.checkpoint
top_k = args.top_k
category_names = args.category_names

#-------------Load Model---------------------
print(f"Loading model from checkpoint {args.checkpoint}")
model = model_utils.load_model(checkpoint)

#--------------Predict------------------------
probs, classes = model_utils.predict(image_path, model, top_k)

#--------------Label mapping------------------------

if category_names:
    with open(category_names, 'r') as f:
        cat_to_name = json.load(f)
    names = [cat_to_name[key] for key in classes]

print("Class number:")
print(classes)
print("Probability (%):")
for idx, item in enumerate(probs):
    probs[idx] = round(item*100, 2)
print(probs)