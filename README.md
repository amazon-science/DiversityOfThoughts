Shield: [![CC BY 4.0][cc-by-shield]][cc-by]

# [ICLR 2025] Enhancing Language Model Agents using Diversity of Thoughts

Official implementation for ICLR 2025 paper [Enhancing Language Model Agents using Diversity of Thoughts](https://openreview.net/forum?id=ZsP3YbYeE9).

Code adapted from [Reflexion](https://github.com/noahshinn024/reflexion/tree/main)

## Setup
1. Install required dependencies into your environment:
```bash
pip install -r requirements.txt
```

2. Set ```OPENAI_API_KEY``` environment key to your OpenAI API Key:
```bash
export OPENAI_API_KEY=<your key>
```

## (optional) Setup for running LeetCodeHardGym experiments:
1. Follow these instructions to prepare the dataset: https://github.com/GammaTauAI/leetcode-hard-gym
2. Login to LeetCode on your browser and retrieve 'csrf' and 'LEETCODEHARD_SESSION' from your browser cookies and set environment variable:
```bash
export LEETCODEHARD_SESSION=<LEETCODE_SESSION str from your browser session>
```
3. Replace csrf_token in line::40 of ```executors/leetcode_env/environment.py```


## Available Agents
* ```dot``` - Diversity of Thoughts Agent
* ```dot_bank``` -- Diversity of Thoughts Agent that uses a Task Agnostic Memory Bank
* ```reflexion``` -- Reflexion Agent
* ```simple``` -- Standard prompting. Set ```max_iters=1```.


## Usage
We provide a sample script ```example_run.sh```. Provide the output directory by changing the value for ```--root_dir``` and run:
```bash
sh example_run.sh
```

* ```--max_iters```: maximum depth of search tree

## Trajectories
We include the trajectories from our paper's experiments in ```Experiments_logs/```

## Cite

```bibtex
@inproceedings{
lingam2025enhancing,
title={Enhancing Language Model Agents using Diversity of Thoughts},
author={Vijay Lingam and Behrooz Omidvar Tehrani and Sujay Sanghavi and Gaurav Gupta and Sayan Ghosh and Linbo Liu and Jun Huan and Anoop Deoras},
booktitle={The Thirteenth International Conference on Learning Representations},
year={2025},
url={https://openreview.net/forum?id=ZsP3YbYeE9}
}
```


This work is licensed under a
[Creative Commons Attribution 4.0 International License][cc-by].

[![CC BY 4.0][cc-by-image]][cc-by]

[cc-by]: http://creativecommons.org/licenses/by/4.0/
[cc-by-image]: https://i.creativecommons.org/l/by/4.0/88x31.png
[cc-by-shield]: https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg


