# Pasma technical demo

25 seconds, 1920×1080, 30 fps. The video shows a single garment moving through local inference, then summarizes model size, native speed and type accuracy.

The interaction is animated from the Pasma demo UI. The example photograph is AI-generated; its displayed predictions are actual results from the published Small-320 FP32 ONNX model, rounded to one decimal place. This is a demonstration input, not a new evaluation or a real-time speed recording.

10.32 MiB is the Small-320 FP16 ONNX file size; the FP32 file is 20.37 MiB. The browser demo uses FP32. The 5.85 ms figure is native CoreML p50 on Mac M2, excluding preparation and loading; phone/browser latency is not represented by that figure.

89.1% vs 80.2% is rounded type Top-1 for task-tuned Pasma Small-320 vs zero-shot SigLIP Base on the same 1,103 held-out real photos, from the saved PyTorch FP32 evaluation. This is not an equal-training comparison. See the [model card and evaluation](https://huggingface.co/Prikol5ist/pasma) for limitations and details.

Media credits: original Pasma branding; imagegen demonstration photo; Happy Beats / Business Moves vol. 12 by ende.app and Kenney sound effects bundled with [latent-spaces/brag](https://github.com/latent-spaces/brag). Video assembled locally with Hyperframes 0.8.191 and the brag workflow. Music and third-party media retain their respective terms; the Apache-2.0 project license describes the model code and fitted weights.
