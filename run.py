
import gradio as gr
import numpy as np
import torch
model = torch.hub.load('ultralytics/yolov5', 'custom', path='last2.pt', force_reload=True)

from transformers import AutoModelForCausalLM, AutoTokenizer
llm = AutoModelForCausalLM.from_pretrained("microsoft/phi-1_5", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-1_5", trust_remote_code=True)

def give(results):
  prompt=""
  if int(results.pred[0][-1][-1].numpy())==0:
      prompt="Suggest a statement for praising my focus."
  elif int(results.pred[0][-1][-1].numpy())==1:
      prompt="Suggest an exercise for staying awake."
  else:
      prompt="Suggest an exercise for staying alert."
  inputs = tokenizer(prompt, return_tensors="pt", return_attention_mask=False)
  outputs = llm.generate(**inputs, max_length=30)
  return tokenizer.batch_decode(outputs)[0]

def detect(im):
    results = model(im)
    return [results,give(results)]
    #return [np.squeeze(results.render())]
    #return [im]

demo = gr.Interface(
    detect,
    [gr.Image(source="webcam", tool=None)],
    ["text","text"],
)
if __name__ == "__main__":
    demo.launch(share=True)
