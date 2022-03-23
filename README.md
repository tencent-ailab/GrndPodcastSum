# Towards Abstractive Grounded Summarization of Podcast Transcripts
We provide the source code for the paper ["Towards Abstractive Grounded Summarization of Podcast Transcripts"](https://arxiv.org/pdf/2203.11425.pdf) accepted at ACL'22. If you find the code useful, please cite the following paper.

    @inproceedings{song-etal-2022-grounded,
        title="Towards Abstractive Grounded Summarization of Podcast Transcripts",
        author = "Song, Kaiqiang and
                  Li, Chen and
                  Wang, Xiaoyang and
                  Yu, Dong and
                  Liu, Fei",
        booktitle={Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics},
        year={2022}
    }

## Goal
We proposed a grounded summarization system, which provide each summary sentence a linked chunk of the original transcripts and their audio/video recordings. It allows a human evaluator to quickly verify the summary content against source clips.
![example](https://raw.githubusercontent.com/tencent-ailab/GrndPodcastSum/main/example.png)

## News
+ 03/22/2022 ArXiv Paper released.
+ 03/04/2022 Trained model and processed testing data released.
+ 03/03/2022 Code Released. Paper link, trained model and processed testing data will be released soon.
+ 02/23/2022 Paper accepted at ACL 2022.

## Experiments

You can follow the below 4 steps to generate grounded podcast summaries or directly download the generated summary from this [link]()
  
## Step 1: Download Code, Model & Data
Download the code
```shell
git clone https://github.com/tencent-ailab/GrndPodcastSum.git
cd GrndPodcastSum
```


Download the [Trained Models](https://tencentamerica-my.sharepoint.com/:u:/p/riversong/EQJXTcDij2tMrxkKq-ezpF8BTXIYxOQlBbI4zJNBSa3_Cg?e=UIb5kU) to ``GrndPodcastSum`` Directory and unzip
```shell
unzip model.zip
```

Download the [Processed Test Set (1027)](https://tencentamerica-my.sharepoint.com/:u:/p/riversong/EUzHYm1Y89NGq5IA4T0f8ygBJ8GWE3EF2nue_umULelN5A?e=AeTDyg) to ``GrndPodcastSum`` Directory and unzip
```shell
unzip data.zip
```

## Step 2: Setup Environment
Create the environment using ``.yml`` file. 
```shell
conda env create -f env.yml
conda activate GrndPodcastSum
```

### Step 3. Offline Computing for Chunk Embeddings
Calculating the chunk embedding offline.
```shell
sh offline.sh
```

### Step 4. Generating Grounded Summary
Use Grnd-token-nonoveralp model to generate summary.
```shell
sh test.sh
```

## License
   Copyright 2022 Tencent

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
   
 ## Disclaimer
 This repo is only for research purpose. It is not an officially supported Tencent product.
