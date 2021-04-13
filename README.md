# Machine Learning Approaches to Historic Music Restoration - Brahms' 1889 Recording
#### Master's thesis for Cal Poly Blended Computer Science Program - [Presentation](https://docs.google.com/presentation/d/10V6d6CxRILrC-cb6raxMNgtBop7NJA5XW9NpZTX9lPc/edit?usp=sharing)

Digital signal processing (pre and post processing) is used in pair with either 2 core machine learning techniques: non-negative matrix factorization (NMF) or deep recurrent neural networks (DRNNs).

Background info, original recording (brahms.wav) & benchmark from [CCRMA Webpage](https://ccrma.stanford.edu/groups/edison/brahms/brahms.html).

Piano samples from [University of Iowa Electronic Music Studios](http://theremin.music.uiowa.edu/MISpiano.html).

#### Restore with NMF (best result):
```
python restore_with_nmf.py brahms.wav
```
#### Restore with DRNN (requires TF >= 2.0, ran on GPU = NVIDIA GTX 970):
```
python restore_with_drnn.py t true
python restore_with_drnn.py r true
```
