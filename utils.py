import torch
import json
from transformers import AutoTokenizer, AutoModel
from torchvision import models, transforms
from PIL import Image
import numpy as np
import re
import pandas as pd
from fusion_model import FusionModel

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load labels
with open("labels.json") as f:
    all_terms = json.load(f)

# Load trained fusion model
fusion_model = FusionModel(out_dim=len(all_terms)).to(DEVICE)
fusion_model.load_state_dict(torch.load("fusion_model.pth", map_location=DEVICE))
fusion_model.eval()

# Load encoders
tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
text_encoder = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT").to(DEVICE)

# Image encoder
class ResNetFeatureExtractor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        base = models.resnet50(pretrained=True)
        self.base = torch.nn.Sequential(*list(base.children())[:-1])
    def forward(self, x):
        return self.base(x).view(x.size(0), -1)

model_image = ResNetFeatureExtractor().to(DEVICE)

# Image preprocessor
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# Load synonym sheet and build type map
synonym_df = pd.read_excel("radiology_vocabulary_final.xlsx", sheet_name="synonyms", engine="openpyxl")
term_type_map = {str(row["Term"]).strip().lower(): str(row["Type"]).strip() for _, row in synonym_df.iterrows()}

# ---- Feature Extractors ----
@torch.no_grad()
def extract_text_features(input_ids, attention_mask):
    return text_encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state.mean(dim=1)

@torch.no_grad()
def extract_image_features(img_tensor):
    return model_image(img_tensor)

# ---- Prediction ----
def predict_gui(text, image):
    try:
        text = text or ""
        has_text = len(text.strip()) > 0
        has_image = image is not None

        print(f"\n📥 Inputs - Text: {has_text}, Image: {has_image}")

        # Text feature
        if has_text:
            tok = tokenizer(text, padding="max_length", truncation=True, max_length=256, return_tensors="pt")
            input_ids = tok["input_ids"].to(DEVICE)
            attn_mask = tok["attention_mask"].to(DEVICE)
            text_feat = extract_text_features(input_ids, attn_mask)
        else:
            text_feat = torch.zeros((1, 768)).to(DEVICE)

        # Image feature
        if has_image:
            if isinstance(image, Image.Image):
                img = image.convert("RGB")
            else:
                img = Image.open(image).convert("RGB")
            img_tensor = transform(img).unsqueeze(0).to(DEVICE)
            img_feat = extract_image_features(img_tensor)
        else:
            img_feat = torch.zeros((1, 2048)).to(DEVICE)

        print(f"🧠 Features — Text: {text_feat.shape}, Image: {img_feat.shape}")

        # Prediction
        with torch.no_grad():
            logits = fusion_model(text_feat, img_feat)
            probs = torch.sigmoid(logits).squeeze().cpu().numpy()

        pred_indices = np.where(probs > 0.5)[0]
        results = []
        for i in pred_indices:
            term = all_terms[i]
            term_type = term_type_map.get(term.lower(), "-")
            results.append({
                "Term": term,
                "Type": term_type,
                "Confidence": f"{probs[i]:.2f}",
                "Source": "Fusion Model"
            })

        highlighted = highlight_text(text, results) if has_text else ""
        return highlighted, pd.DataFrame(results)

    except Exception as e:
        import traceback
        print("🚨 Error in predict_gui:", e)
        traceback.print_exc()
        return "Prediction Failed", pd.DataFrame(columns=["Term", "Type", "Confidence", "Source"])

# ---- Highlighting Function ----
def highlight_text(text, results):
    ranges = []
    for r in results:
        term = r["Term"]
        for match in re.finditer(re.escape(term), text, re.IGNORECASE):
            ranges.append({"start": match.start(), "end": match.end(), "entity": term})
    return {"text": text, "entities": ranges}
