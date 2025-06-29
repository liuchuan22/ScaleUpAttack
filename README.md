# Scaling Laws for Black-box Adversarial Attacks

> We investigate the scaling laws of black-box adversarial attacks in this work. Through theoretical analysis and empirical evaluations, we conclude with clear scaling laws that using more surrogate models enhances adversarial transferability. Comprehensive experiments verify the claims on standard image classifiers, diverse defended models and multimodal large language models using various adversarial attack methods. Specifically, by scaling law, we achieve 90%+ transfer attack success rate on even proprietary models like GPT-4o.

For details please refer to our paper [here](https://arxiv.org/abs/2411.16782).

## Data Preparation

For NIPS17 dataset, you can run

    kaggle datasets download -d google-brain/nips-2017-adversarial-learning-development-set

and then place it in `./resources/NIPS17`.

We provide several target images for demonstration in `./resources/cifar10`.

## Usage

We provide some example code for reproducing our experiments in the paper.

**scale_up_attack_classifier.py**: Training adversarial examples using an ensemble of image classifiers. 

**scale_up_attack_clips.py**: Training adversarial examples using an ensemble of clips. 

You can set the target image, ensemble size, attack algorithm, result saving path, and training settings in the code script.

**eval_classifiers.py**: Record the evaluation loss and attack success rate over a set of image classifiers.

**eval_defense.py**: Record the attack success rate over a list of defense methods.

**eval_vllms/eval_vllms.py**: Evaluate the attack success rate over popular VLLMs. If you would like to evaluate VLLM performance under [Multitrust](https://arxiv.org/abs/2406.07057) framework with our adversarial examples, please replace our crafted adversarial examples with data in its `robustness/r5-adversarial-target` section. We provide configs in `data/config` folder.

## Citation
Please cite us:

```
@misc{liu2025scalinglawsblackbox,
      title={Scaling Laws for Black box Adversarial Attacks}, 
      author={Chuan Liu and Huanran Chen and Yichi Zhang and Yinpeng Dong and Jun Zhu},
      year={2025},
      eprint={2411.16782},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2411.16782}, 
}
```

If you have any question, you can contact us by:  

Email: liuchuan22@mails.tsinghua.edu.cn
