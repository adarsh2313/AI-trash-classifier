import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from model import model

def predict_image(img, model):
    transformations = transforms.Compose([transforms.Resize((300, 300)), 
                                      transforms.ToTensor(),
                                      transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]),])

    img = transformations(img)
    img = img.unsqueeze(0)
    model.eval()
    classes = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']
    out = model(img)
    prob, preds  = torch.max(out, dim=1)
    answer = classes[preds[0].item()]
    confidence = round(100*prob.item(),4)
    return (answer,confidence)