
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


scp $LOCAL/datasets/FB-HM/data/ocr-fbhm.json $ADA/datasets/FB-HM/data

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





#### Debugging errors
If you already have done the above, then the distributed data parallel module wasn't able to locate the output tensors in the return value of your module's `forward` function. Please include the loss function and the structure of the return value of `forward` of your module when reporting this issue (e.g. list, dict, iterable).
Parameter indices which did not receive grad for rank 1: 0 198 199 200 201 203 399 400 401 402 504 505 506 507

https://github.com/huggingface/accelerate/issues/24

https://discuss.huggingface.co/t/runtimeerror-expected-to-have-finished-reduction-in-the-prior-iteration-before-starting-a-new-one-this-error-indicates-that-your-module-has-parameters-that-were-not-used-in-producing-loss/64760
