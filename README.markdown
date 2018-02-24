
This is a project use seq2seq model to play couplets (对对联)。 This project is written with Tensorflow. You can try the demo at [https://ai.binwang.me/couplet](https://ai.binwang.me/couplet).


Dataset
-----------

You will need some data to run this program, the dataset can be downloaded from [this project](https://github.com/wb14123/couplet-dataset).

** Note: If you are using your own dataset, you need to add `<s>` and `<\s>` as the first two line into the vocabs file.

Usage
------------

### Train

Open `couplet.py` and config the file locations and hyperparams. Then run `python couplet.py` to train the model. You can see the training loss and bleu score at Tensorbloard. You may need to re-config `learning_rate` when you find the loss stops descresing. Here is an example of the loss graph:

If you stoped the training and want to continue to train it. You can set `restore_model` to `True` and use `m.train(<epoches>, start=<start>)`, which `start` is the steps you've already run.


### Run the trained model

Open `server.py` and config the `vocab_file` and `model_dir` params. Then run `python server.py` will start a web service that can play couplet.
