# Defending Large Language Models Against Jailbreak Attacks via Layer-specific Editing

This repository contains the code for the paper "Defending Large Language Models Against Jailbreak Attacks via Layer-specific Editing". The focus of this work is to enhance the resilience of large language models (LLMs) against jailbreak attacks through a novel method termed Layer-specific Editing (LED).

## Table of Contents
- [Introduction](#introduction)
- [Installation](#installation)
- [Usage](#usage)
  - [Pruning Analysis](#pruning-analysis)
  - [Hidden Analysis](#hidden-analysis)
- [Pre-trained Models](#pre-trained-models)
- [Citation](#citation)
- [License](#license)

## Introduction

Large language models (LLMs) such as GPT-4, Llama2, Vicuna, and Mistral have demonstrated remarkable capabilities across various natural language tasks. However, these models are vulnerable to adversarial prompts, also known as jailbreak attacks, which can elicit harmful or unintended behaviors. Our proposed method, Layer-specific Editing (LED), aims to enhance the model's defenses against such attacks by focusing on specific layers within the model.

## Installation

To get started, clone this repository and install the required dependencies:

```bash
git clone https://github.com/ledllm/ledllm
cd ledllm
pip install -r requirements.txt
```

## Usage
### Pruning Analysis
The pruning analysis identifies crucial layers in the model that contribute significantly to its defense against harmful prompts. To run the pruning analysis, use the following command:
```bash
python pruning_analysis.py
```

### Hidden Analysis
The hidden analysis decodes hidden states into vocabulary space to observe the probability of each decoded token. This helps in identifying layers that retain a high probability of decoding refusal tokens. To run the hidden analysis, use the following command:

```bash
python hidden_states_analysis.py
```

