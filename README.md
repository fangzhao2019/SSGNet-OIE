# SSGNet-OIE
For paper:《Neural Open Information Extraction with Set Sequence Generation Networks》

## Execution Instructions
### Data Download
bash download_data.sh 

This downloads the (train, dev, test) data

### Running the code
```
python allennlp_script.py --param_path spnie/configs/spnie.json --s models/spnie --mode train_test
```

Arguments:
- param_path: file containing all the parameters for the model
- s:  path of the directory where the model will be saved
- mode: train, test, train_test
