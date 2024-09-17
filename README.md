<div style="text-align: center;">
    <img src="https://github.com/mnabihali/FL-WavLM-with-Adapters/blob/main/assets/ff.webp" alt="EFL-PEFT Banner" />
</div>

# Description
This repo. contains our implementation for our research "EFL-PEFT: A communication Efficient Federated Learning framework using PEFT sparsification for ASR". The paper is submitted to ICASSP (2025). More details on how to use the code, and advanced scripts will be available soon. Stay tuned...

# EFL-PEFT architecture
<img src="https://github.com/mnabihali/WavLM_Adapters_FL/blob/main/assets/FL.png" width="512"/>

# WavLM + EL adapters
<img src="https://github.com/mnabihali/WavLM_Adapters_FL/blob/main/assets/EL.png" width="512"/>

# How to use
### Dataset preparation
--> Will add it asap <--
### Train
Syntax: To use EL adapters `python client.py --train_lawithea true`
### Inference
Syntax: To use EL adapters `python inference.py --train_lawithea true`


# References
Our implementation is based on this nice work:
```
@inproceedings{otake2023parameter,
  title = {Parameter Efficient Transfer Learning for Various Speech Processing Tasks},
  author = {S. Otake, R. Kawakami, N. Inoue},
  booktitle = {Proc. ICASSP},
  year = {2023},
  code_url = {https://github.com/sinhat98/adapter-wavlm}
}
```

# Publication
Our research paper is under review at ICASSP 2025. It will be on arxiv soon.
```
@inproceedings{mnabihali,
  title = {EFL-PEFT: A communication Efficient Federated Learning framework using PEFT sparsification for ASR},
  author = {M.Nabih, D. Falavigna, A. Brutti},
  booktitle = {Under review ICASSP},
  year = {2025},
}
```


