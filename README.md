---
title: IntegraBNER
emoji: 🧠
colorFrom: gray
colorTo: indigo
sdk: gradio
sdk_version:  5.24.0
app_file: app.py
pinned: false
---

# IntegraBNER - Multimodal Biomedical Named Entity Recognition

Upload a radiology report and/or X-ray image. IntegraBNER highlights relevant biomedical terms using a text+image fusion model. Built with Bio_ClinicalBERT, ResNet50, and hybrid autolabeling (dictionary + BioBERT + LLM).

🔍 Fine-tuned on OpenI + synthetic biomedical labels  
🧠 Fusion Model: Attention-enhanced MLP  
📊 Outputs: Predicted Terms, Types, Confidence, and Source  
🖼️ Gradio App powered by Hugging Face Spaces
