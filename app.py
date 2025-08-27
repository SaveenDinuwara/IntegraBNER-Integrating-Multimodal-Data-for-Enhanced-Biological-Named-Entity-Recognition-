import gradio as gr
from utils import predict_gui, all_terms, highlight_text

def build_interface():
    return gr.Interface(
        fn=predict_gui,
        inputs=[
            gr.Textbox(
                label="📝 Radiology Report",
                placeholder="e.g., Cardiomegaly and pulmonary markings..."
            ),
            gr.Image(
                type="pil",
                label="🩻 X-ray Image (Optional)"
            )
        ],
        outputs=[
            gr.HighlightedText(
                label="🧠 Highlighted Report",
                combine_adjacent=True
            ),
            gr.Dataframe(
                headers=["Term", "Type", "Confidence", "Source"],
                label="📋 Predicted Biomedical Terms"
            )
        ],
        title="🧠 IntegraBNER – Multimodal Biomedical Named Entity Recognition",
        description=(
            "This system extracts relevant biomedical entities from radiology reports and X-ray images using a multimodal fusion model.\n\n"
            "**Entity Type Codes:**  \n"
            "`A` = Anatomical Structure  `D` = Disease or Pathology  `S` = Sign/Symptom/Finding  `-` = Unspecified"
        ),
        allow_flagging="never"
    )

if __name__ == "__main__":
    interface = build_interface()
    interface.launch(debug=True, share=True)
