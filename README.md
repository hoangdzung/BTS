# BTS
BerTStorypoint 
Story point esimation using pretrained BERT and Linear Regression

## Installation

### Requirements
Code is tested on:
  * Python 3.7.4
  * Linux
  * Numpy 1.17.2
  * Pytorch 1.3.1
  * Keras 1.3.1
  * Transformers 2.1.1
  * scikit-learn 0.21.3
  * Tqdm 
  
### Installation Options:

#### Installing on Mac or Linux
  * Install pytorch: Go to [official pytorch page](https://pytorch.org/) to get the install command which's compatiable with your machine. If current version is larger than 1.3, go to [previous version](https://pytorch.org/get-started/previous-versions/) to get the wanted version.

If you have a GPU:
```bash
pip3 install torch torchvision
```
And without GPU:
```bash
pip3 install torch==1.3.1+cpu torchvision==0.4.2+cpu -f https://download.pytorch.org/whl/torch_stable.html
```
  * For numpy, keras, scikit-learn, tqdm, you can easily install by using `pip3` (or `pip`):

```bash
pip3 install numpy keras scikit-learn tqdm transformers
```
Most of time, the command above is enough, but sometime you'll get a version error. To solve it, please uninstall the wrong-version package, then re-install them with version specified:
```bash
pip3 install keras==1.3.1 scikit-learn==0.21.3 transformers==1.3.1
```

## Usage
Example:
``` bash
python3 main.py --datafile data-file --dictfile dict-file --dropout 0.5 --batch_size 16 --epochs 50 --hidden_size 128
```
* `data-file` and `dict-file` are in `data/processed/` directory. `data-file` ends with `.pkl.gz` and `dict-file` ends with `.dict.pkl.gz`
They must come with pair:

| Project name | data-file         | dict-file    |
| :----:       |    :----          |        :---- |
| ME           | mesos             | apache       | 
| UG           | usergrid          | apache       |
| AS           | appceleratorstudio| appcelerator |
| AP           | aptanastudio      | appcelerator |
| TI           | titanium          | appcelerator |
| DC           | duracloud         | duraspace    | 
| BB           | bamboo            | jira         |
| CV           | clover            | jira         |
| JI           | jirasoftware      | jira         |
| MD           | moodle            | moodle       |
| DM           | datamanagement    | lsstcorp     |
| MU           | mulestudio        | mulesoft     |
| MS           | mule              | mulesoft     |
| XD           | springxd          | spring       |
| TD           | talenddataquality | talendforge  |
| TE           | talendesb         | talendforge  |

    
For see all argument, run:
 ``` bash
python3 main.py --help
```
