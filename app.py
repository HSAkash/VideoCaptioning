import time
from src.config.configuration import ConfigurationManager
from src.components.refineGeneratedCaption import RefineGeneratedCaption
from src.components.generateCaption import GenerateCaption
from pyprojroot import here
import gradio as gr
import os

# Initialize components
configurationManager = ConfigurationManager()
gen_config = configurationManager.get_generate_caption_config()
refine_config = configurationManager.get_refine_generated_caption_config()

generateCaption = GenerateCaption(gen_config)
refineGeneratedCaption = RefineGeneratedCaption(refine_config)

model_path = here("dataset/MSR-VTT/MSRVTT_checkpoints/train_model")

def process_video_with_captions(video_file):
    """Process video and generate captions - returns captions immediately"""
    if video_file is None:
        return "Ready for upload", "", ""
    
    video_path = video_file.name if hasattr(video_file, 'name') else video_file
    
    if not os.path.exists(video_path):
        return "File not found", "", ""
    
    try:
        # Generate caption
        gen_cap = generateCaption.generate(video_path, model_path)
        # Refine caption
        refine_cap = refineGeneratedCaption.run(caption=gen_cap)
        print(refine_cap)
        
        return f"✅ Caption generated for: {os.path.basename(video_path)}", gen_cap, refine_cap
        
    except Exception as e:
        return f"❌ Error: {str(e)}", "", ""

# Create interface
with gr.Blocks(title="AI Video Caption Generator", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🎬 AI Video Caption Generator")
    gr.Markdown("Upload a video to generate captions")
    
    with gr.Row():
        with gr.Column(scale=1):
            video_input = gr.Video(
                label="📹 Upload/Play Video",
                sources=["upload"],
                include_audio=True,
                height=300
            )
            process_btn = gr.Button("🚀 Generate Captions", variant="primary")
            
        with gr.Column(scale=1):
            status = gr.Textbox(
                label="Status",
                interactive=False,
                value="Upload a video and click Generate"
            )
            generated_caption = gr.Textbox(
                label="Generated Caption",
                lines=4,
                interactive=False
            )
            refined_caption = gr.Textbox(
                label="Refined Caption",
                lines=4,
                interactive=False
            )
    
    # Connect button - only process captions, not video
    process_btn.click(
        fn=process_video_with_captions,
        inputs=video_input,
        outputs=[status, generated_caption, refined_caption]
    )
    
    # Optional: Auto-generate when video is uploaded
    # video_input.change(
    #     fn=process_video_with_captions,
    #     inputs=video_input,
    #     outputs=[status, generated_caption, refined_caption]
    # )

if __name__ == "__main__":
    demo.launch(debug=True)