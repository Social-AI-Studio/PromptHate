# Prompt-Based Hateful Meme Detection

This is the implementation of the paper Prompting for Multimodal Hateful Meme Classification (EMNLP 2022).

## Overview

![](imgs/prompthate.JPG)

Hateful meme classification is a challenging multimodal task that requires complex reasoning and contextual background knowledge. 
Ideally, we could leverage an explicit external knowledge base to supplement contextual and cultural information in hateful memes. 
However, there is no known explicit external knowledge base that could provide such hate speech contextual information. 
To address this gap, we propose PromptHate, a simple yet effective prompt-based model that prompts pre-trained language models (PLMs) for hateful meme
classification. Specifically, we construct simple prompts and provide a few in-context examples to exploit the implicit knowledge in the pre-
trained RoBERTa language model for hatefulmeme classification.

## Data Pre-processing
In order to run the code, you need to pre-process data in the following step:

1. clean meme texts of images.
2. generate captions over each cleaned image.
3. extract entity and demographic information for each cleaned image.

**Image Cleaning**: This step removes the meme texts of images. We follow the implementation provided in the [project](https://github.com/HimariO/HatefulMemesChallenge), which use OCR detection tools to detect texts of image first, remove and impaint the image.

**Caption Generation**: We leverage a pre-trained [image caption generation tool](https://github.com/rmokady/CLIP_prefix_caption). To run the code, you need to download the pre-trained models provided by the project and execute codes in `captions-for-hatefulmeme.ipynb`. Noted, generated captions over cleaned and uncleaned images are obviously different. Specifically, we use the caption generator pre-trained on Conceptual Caption and generate captions over cleaned images.

**Entity and Demographic Information Extraction**: We use Google Vision for entity detection and FairFace detection for demographic information extraction. More details can be found in the [project](https://github.com/HimariO/HatefulMemesChallenge).

We also provide the pre-processed data in the `data/domain_splits` and `caption` folders.

## Run PromptHate
Our code is built on [transformers](https://github.com/huggingface/transformers) and we use its `4.19.2` version and PyTorch using CUDA version 10.2 (compatiable with other versions of transformers, pytorch and CUDA, but may results in unexpected errors). It takes up 19 GB dedicated memory for PromptHate training.

To replicate the reported performance, you can simply run:
```bash
bash run.sh
```

## Citation

Please cite our paper if you use PromptHate in your work:

```bibtex
@inproceedings{rui2022prompthateemnlp,
   title={Prompt-Based Hateful Meme Detection},
   author={Rui Cao, Roy Ka-Wei Lee, Wen-Haw Chong, Jing Jiang},
   booktitle={Empirical Methods in Natural Language Processing (EMNLP)},
   year={2022}
}
```

Part of our work is based on the following works:

```bibtex
@inproceedings{gao2021making,
   title={Making Pre-trained Language Models Better Few-shot Learners},
   author={Gao, Tianyu and Fisch, Adam and Chen, Danqi},
   booktitle={Association for Computational Linguistics (ACL)},
   year={2021}
}
```

```bibtex
@article{mokady2021clipcap,
  title={ClipCap: CLIP Prefix for Image Captioning},
  author={Mokady, Ron and Hertz, Amir and Bermano, Amit H},
  journal={arXiv preprint arXiv:2111.09734},
  year={2021}
}
```
