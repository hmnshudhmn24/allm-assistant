import gradio as gr
from src.model import AllMAssistant

# Use the hub model id if you have uploaded (e.g. 'username/AllM-Assistant')
MODEL_SOURCE = 'gpt2'  # replace with 'your-username/AllM-Assistant' after upload

assistant = AllMAssistant(MODEL_SOURCE)

def generate_handler(text: str):
    prompt = text.strip()
    if not prompt:
        return "Please enter an instruction."
    return assistant.generate(prompt, max_new_tokens=250)

with gr.Blocks() as demo:
    gr.Markdown("# AllM-Assistant — Demo\nEnter an instruction like: 'Create a 15-min HIIT for beginners'" )
    txt = gr.Textbox(lines=4, placeholder='Instruction + input (optional)')
    out = gr.Textbox(lines=12)
    btn = gr.Button('Generate')
    btn.click(generate_handler, inputs=txt, outputs=out)

if __name__ == '__main__':
    demo.launch()
