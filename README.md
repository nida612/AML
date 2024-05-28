# bengali_grapheme_classification
OCR for Bengali handwriting with non linear positioning: Solving a multi-target classification problem having three targets (grapheme root, consonant diacritic,vowel diacritic) by using Resnet and EffecientNet as our base models where we further train over our specific dataset.

## steps to run inference:
- Download competition data to folder kaggle/input from https://www.kaggle.com/competitions/bengaliai-cv19/data
- ```python main.py -op test -m resnet34``` or ```python main.py -op test -m efficientnet-b3```
- This will generate **submission.csv** which can be submitted to the competition for evaluation

All available run commands are present under ```.idea/runConfigurations/```
