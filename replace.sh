echo 'install detectron2'
# python3 -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
sudo rm -rf ../detectron2/detectron2/utils/events.py
sudo cp -r ./events.py ../detectron2/detectron2/utils/


sudo rm -rf ../detectron2/detectron2/utils/events.py
sudo cp -r ./events.py ../detectron2/detectron2/utils/
