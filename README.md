# bengali_grapheme_classification
## steps to run inference:
- Download competition data to folder kaggle/input from https://www.kaggle.com/competitions/bengaliai-cv19/data
- ```python main.py -op test -m resnet34``` or ```python main.py -op test -m efficientnet-b3```
- This will generate **submission.csv** which can be submitted to the competition for evaluation

All available run commands are present under ```.idea/runConfigurations/```