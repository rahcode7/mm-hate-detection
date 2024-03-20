
##### Move files
export LOCAL=/Users/rahulmehta/Desktop/Research24/Challenges/MMHate
export ADA=rahul.mehta@ada:mm-hate-detection
cp $LOCAL/models/flava/flava-train.py $LOCAL/mm-hate-detection/models/flava/flava-train.py 
cp $LOCAL/models/flava/ocr-extract.py $LOCAL/mm-hate-detection/models/flava/ocr-extract.py



scp -r /Users/rahulmehta/Desktop/Research24/Challenges/MMHate/datasets/FB-HM/archive.zip rahul.mehta@ada:mm-hate-detection/datasets/FB-HM
scp $LOCAL/mm-hate-detection/models/flava/ada-script.sh $ADA/models/flava
scp $LOCAL/mm-hate-detection/models/flava/flava-train.py $ADA/models/flava
scp $LOCAL/datasets/results/FB-HM/ocr-fbhm.json $ADA/datasets/FB-HM/input-text
scp $LOCAL/mm-hate-detection/models/flava/MM_data_loader.py $ADA/models/flava/MM_data_loader.py


scp $LOCAL/datasets/FB-HM/data/test.jsonl $ADA/datasets/FB-HM/data
scp $LOCAL/datasets/FB-HM/data/train.jsonl $ADA/datasets/FB-HM/data
scp $LOCAL/datasets/FB-HM/data/dev.jsonl $ADA/datasets/FB-HM/data

##### Install packages
python -m pip install torchmultimodal-nightly


##### OCR
pip install easyocr

import easyocr
reader = easyocr.Reader(['ch_sim','en'])
result = reader.readtext('Examples/chinese.jpg')


#### FLAVA Run
sbatch models/flava/ada-script.sh 
squeue -u $USER
cat runs/flava/flava.txt
