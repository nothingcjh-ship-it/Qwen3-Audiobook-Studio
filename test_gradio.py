import gradio as gr

def greet(name):
    return "Hello " + name + "!"

demo = gr.Interface(fn=greet, inputs="text", outputs="text")
print("Launching Gradio test...")
demo.launch(server_name="127.0.0.1", server_port=8000)
