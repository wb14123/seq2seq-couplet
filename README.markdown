
This is a project use seq2seq model to play couplets (对对联)。 This project is written with Tensorflow. You can try the demo at [https://ai.binwang.me/couplet](https://ai.binwang.me/couplet).

Pre-requirements
--------------

* Tensorflow
* Python 3.6
* Dataset


Dataset
-----------

You will need some data to run this program, the dataset can be downloaded from [this project](https://github.com/wb14123/couplet-dataset).

** Note: If you are using your own dataset, you need to add `<s>` and `<\s>` as the first two line into the vocabs file. **

Usage
------------

### Train

Open `couplet.py` and config the file locations and hyperparams. Then run `python couplet.py` to train the model. You can see the training loss and bleu score at Tensorbloard. You may need to re-config `learning_rate` when you find the loss stops descresing. Here is an example of the loss graph:

![loss graph](https://user-images.githubusercontent.com/1906051/36624881-50586e54-1950-11e8-8383-232763831cbc.png)

If you stoped the training and want to continue to train it. You can set `restore_model` to `True` and use `m.train(<epoches>, start=<start>)`, which `start` is the steps you've already run.

I've trained the model on a Nivida GTX-1080 GPU for about 4 days.


### Run the trained model

Open `server.py` and config the `vocab_file` and `model_dir` params. Then run `python server.py` will start a web service that can play couplet.
