
##### Move files
export LOCAL=/Users/rahulmehta/Desktop/Research24/Challenges/MMHate
export ADA=rahul.mehta@ada:mm-hate-detection
cp $LOCAL/models/flava/flava-train.py $LOCAL/mm-hate-detection/models/flava/flava-train.py 
cp $LOCAL/models/flava/ocr-extract.py $LOCAL/mm-hate-detection/models/flava/ocr-extract.py


##### Image only models
scp -r /Users/rahulmehta/Desktop/Research24/Challenges/MMHate/datasets/FB-HM/archive.zip rahul.mehta@ada:mm-hate-detection/datasets/FB-HM
scp $LOCAL/mm-hate-detection/models/flava/ada-script.sh $ADA/models/flava
scp $LOCAL/mm-hate-detection/models/flava/flava-train.py $ADA/models/flava
scp $LOCAL/datasets/results/FB-HM/ocr-fbhm.json $ADA/datasets/FB-HM/input-text
scp $LOCAL/mm-hate-detection/models/flava/MM_data_loader.py $ADA/models/flava/MM_data_loader.py


##### datasets
scp $LOCAL/datasets/FB-HM/data/test.jsonl $ADA/datasets/FB-HM/data
scp $LOCAL/datasets/FB-HM/data/train.jsonl $ADA/datasets/FB-HM/data
scp $LOCAL/datasets/FB-HM/data/dev.jsonl $ADA/datasets/FB-HM/data


##### OCR model
scp $LOCAL/datasets/FB-HM/data/ocr-fbhm.json $ADA/datasets/FB-HM/data
scp $LOCAL/mm-hate-detection/models/flava/MM_data_loader_ocr.py $ADA/models/flava
scp $LOCAL/mm-hate-detection/models/flava/flava-train-ocr.py $ADA/models/flava

scp $LOCAL/mm-hate-detection/models/flava/main.py $ADA/models/flava

scp $ADA/models/
<!-- /Users/rahulmehta/Desktop/Research24/Challenges/MMHate/mm-hate-detection/models/flava/mm-train-ocr.py -->


scp -r Users/rahulmehta/Desktop/Research24/Challenges/MMHate/AISG-Online-Safety-Challenge-Submission-Guide/local_test/



test_images $ADA/datasets
##### Install packages
python -m pip install torchmultimodal-nightly
pip install pytesseract

##### OCR
pip install easyocr

import easyocr
reader = easyocr.Reader(['ch_sim','en'])
result = reader.readtext('Examples/chinese.jpg')


#### Model 1 - FLAVA
sbatch models/flava/ada-script.sh 
squeue -u $USER
cat runs/flava/flava.txt

###### Inference
python models/flava/main.py 
test_images/8b52c3.png
test_images/8b52el.png


#### Model 2 - LLAVA 
sbatch models/flava/ada-script.sh 
squeue -u $USER
cat runs/llava/llava.txt

### OCR Pytesseract

TESSDATA_DIR="/opt/local/share/tessdata/"

# Download English traineddata
sudo wget -P "$TESSDATA_DIR" https://github.com/tesseract-ocr/tessdata/raw/main/eng.traineddata

# Download Simplified Chinese traineddata
sudo wget -P "$TESSDATA_DIR" https://github.com/tesseract-ocr/tessdata/raw/main/chi_sim.traineddata

# Download Traditional Chinese traineddata
sudo wget -P "$TESSDATA_DIR" https://github.com/tesseract-ocr/tessdata/raw/main/chi_tra.traineddata

# Download Tamil traineddata
sudo wget -P "$TESSDATA_DIR" https://github.com/tesseract-ocr/tessdata/raw/main/tam.traineddata

# Download Malay trainneddata
sudo wget -P "$TESSDATA_DIR" https://github.com/tesseract-ocr/tessdata/raw/main/msa.traineddata 


sudo wget -P "$TESSDATA_DIR" https://github.com/tesseract-ocr/tessdata/raw/main/osd.traineddata 



export SUBMISSION_ID=submission2 


# EasyOCR donwload
cp ~/.EasyOCR/model/english_g2.pth $LOCAL/submissions/$SUBMISSION_ID/.cache
cp ~/.EasyOCR/model/zh_sim_g2.pth $LOCAL/submissions/$SUBMISSION_ID/.cache

## Docker  
###### 1. Get requirements

cd submissions/$SUBMISSION_ID
pip install pipreqs 
pipreqs . 

###### 2. Get trained checkpoint file
mkdir $SUBMISSION_ID/checkpoints

scp -r $ADA/checkpoints/checkpoints-flava-ddp-ocr-dev-29Mar $LOCAL/checkpoints

###### 3. Move main.py to submission
cp $ADA/mm-hate-detection/models/flava/main.py $LOCAL/submissions/submission1
cp -r $ADA/checkpoints $LOCAL/submissions/submission1

###### 4 Get models HF

mkdir SUBMISSION_ID/.cache
cp -r /Users/rahulmehta/.cache/huggingface/hub/models--facebook--flava-full /Users/rahulmehta/Desktop/Research24/Challenges/MMHate/submissions/SUBMISSION_ID/.cache

cp -r /Users/rahulmehta/.cache/huggingface/hub/models--facebook--m2m100_418M /Users/rahulmehta/Desktop/Research24/Challenges/MMHate/submissions/SUBMISSION_ID/.cache


###### 3. DOCKER build image from submission folder

docker build -t $SUBMISSION_ID .
docker save $SUBMISSION_ID | gzip > $SUBMISSION_ID.tar.gz


# Test docker 

docker network create \
    --driver "$ISOLATED_DOCKER_NETWORK_DRIVER" \
    $( [ "$ISOLATED_DOCKER_NETWORK_INTERNAL" = "true" ] && echo "--internal" ) \
    --subnet "$ISOLATED_DOCKER_NETWORK_SUBNET" \
    "$ISOLATED_DOCKER_NETWORK_NAME"

docker system prune -a


ISOLATED_DOCKER_NETWORK_NAME=exec_env_jail_network
cat local_test/test_stdin/stdin.csv | \
docker run --init \
        --attach "stdin" \
        --attach "stdout" \
        --attach "stderr" \
        --cpus 1 \
        --memory 4g \
        --memory-swap 0 \
        --ulimit nproc=1024 \
        --ulimit nofile=1024 \
        --network exec_env_jail_network \
        --read-only \
        --mount type=bind,source="$(pwd)"/local_test/test_images,target=/images,readonly \
        --mount type=tmpfs,destination=/tmp,tmpfs-size=5368709120,tmpfs-mode=1777 \
        --interactive \
        submission2 \
 1>local_test/test_output/stdout.csv \
 2>local_test/test_output/stderr.csv

#### Debugging errors
If you already have done the above, then the distributed data parallel module wasn't able to locate the output tensors in the return value of your module's `forward` function. Please include the loss function and the structure of the return value of `forward` of your module when reporting this issue (e.g. list, dict, iterable).
Parameter indices which did not receive grad for rank 1: 0 198 199 200 201 203 399 400 401 402 504 505 506 507

https://github.com/huggingface/accelerate/issues/24

https://discuss.huggingface.co/t/runtimeerror-expected-to-have-finished-reduction-in-the-prior-iteration-before-starting-a-new-one-this-error-indicates-that-your-module-has-parameters-that-were-not-used-in-producing-loss/64760
