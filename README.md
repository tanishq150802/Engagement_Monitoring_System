# Engagement_Monitoring_System
### LLM &amp; Yolov5 Powered Engagement Monitoring System for Online learning management.

By: [Tanishq Selot](https://github.com/tanishq150802)

Deployed at [Huggingface Spaces](https://huggingface.co/spaces/tanishq1508/LLM_based_Engagement_lvl_alert_system/tree/main). Takes ```16 s``` with ```CPU```.

Use ```run.py``` to run the gradio app. The yolov5 model is stored within ```last2.pt```. ```Yolov5_finetuning.ipynb``` is used for finetuning the Yolov5 architecture for "drowsy", "looking away" and "awake" classes.

## Flow
* Finetuned Yolov5 is used to detect the engagement class from webcam.
* The detected class is sent to [Phi-1.5](https://huggingface.co/microsoft/phi-1_5), a 1.3B param LLM to generate the alert message.

## Examples

  Awake             |  Looking Away |  Drowsy
:-------------------------:|:-------------------------: |:-------------------------:
![awake](https://github.com/tanishq150802/Engagement_Monitoring_System/assets/81608921/72f2f2d5-cc03-4316-9caa-0d1609796bd5) |  ![looking_away](https://github.com/tanishq150802/Engagement_Monitoring_System/assets/81608921/7c17e207-8a96-4d24-b7a2-d723b2aa908d) |  ![drowsy](https://github.com/tanishq150802/Engagement_Monitoring_System/assets/81608921/d7bd99cf-ede4-4ca5-a8b6-5ad9846ca47c)


## Requirements
* streamlit==0.84.1
* Pillow
* jax[cpu]
* flax
* transformers
* git+https://github.com/huggingface/transformers
* huggingface_hub
* googletrans==4.0.0-rc1
* protobuf==3.20
* torch
* gradio
* numpy
* opencv-python
* einops

## References
* [Microsoft's Phi-1.5](https://huggingface.co/microsoft/phi-1_5)
* [Webcam spaces](https://huggingface.co/spaces/gradio/webcam)
