# MM-office dataset
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.6088572.svg)](https://doi.org/10.5281/zenodo.6088572)

MM-office is a multi-view and multi-modal dataset in an office environment (MM-Office) that records events, e.g., 'enter' to the office room, 'sit down' on the chair, and 'take out' something from a shelf, in the room assuming the daily work. These events are recorded simultaneously using eight non-directional microphones and four cameras. The audio and video clips are divided into scenes, each about 30 to 90 seconds. The amount of data was 880 clips per point and sensor. The labels available for training are given as multi-labels that indicate which each clip contains what event. Only the test data is annotated with a strong label containing the onset/offset time of each event.

## Download

You can download the dataset [here](https://doi.org/10.5281/zenodo.6088572). 

## Details of dataset
The dataset has following folder structure:

<pre>
MM_Office_Dataset
├── audio
│   ├── test
│   └── train
├── video
│   ├── test
│   └── train
└──label
    ├── testlabel
    └── trainlabel
        ├── classinfo.csv
        ├── eventinfo.csv
        └── recinfo.csv
</pre>


### audio/video
Audio and video were recorded synchronously using four cameras (_GoPro HERO8_) and eight non-directional microphones (_HOSIDEN KUB4225_) installed in the office, as shown in the room setup figure below. The audio was recorded at 48kHz/16bit. The video was recorded at 1920×1440/30fps, and then resized to 480

![roomsetup](https://user-images.githubusercontent.com/72438001/154604637-bebad048-1ba3-40be-92de-951b6f0b770a.png)

The naming convention for these recordings is as follows. 
<pre>
split[split index]_id[sensor index]_s[scene index]_recid[recording id]_[division].[wav or mp4]
</pre>
The MM-Office dataset is split into 10 splits for convenience, and the _split index_ (0 to 9) is the index of that. The _sensor index_ is the sensor number of the camera and microphone and corresponds to the room setup figure above (but starts with 0). The _scene index_ is an index that shows the scenario pattern of actions performed by the actors. Refer to ```eventinfo.csv``` to see what kind of actions and events each scene contains. The _recording id_ is the serial number of the recording, but after recording, we decided to split each recording in half to make a single clip, so each recording id has two duplicates. The _division_ indicates this, where the first half is 0 and the second half is 1.

### label
#### testlabel

| index | eventclass | starttime | endtime |
|-------|------------|-----------|---------|
| 0     | 8          | 6         | 14      |
| 1     | 11         | 20        | 35      |

#### recinfo.csv

| recid | sceneid | patternid |
|-------|---------|-----------|
| 0     | 1       | 1         |
| ...   |         |           |
| 679   | 11      | 1         |

#### eventinfo.csv

| sceneid | patternid | division | class1 | class2 | class3 | class4 | ... | class12 |
|---------|-----------|----------|--------|--------|--------|--------|-----|---------|
| 5       | 1         | 0        | 0      | 0      | 0      | 1      |     | 0       |
| ...     |           |          |        |        |        |        |...     |         |
| 3       | 6         | 1        | 0      | 1      | 0      | 0      |     | 1       |

#### classinfo.csv 
It contains the event name (e.g. 'stand up,' 'phone') of each of the event classes shown in ```eventinfo.csv``` and a description of what kind of event it is.

## Usage
We plan to release a DataLoader for PyTorch soon.

## License
See this [license](./LICENSE.pdf) file.

## Authors and Contact
        
* Masahiro Yasuda (Email: masahiro.yasuda@ieee.org)
* Yasunori Ohishi
* Shoichiro Saito
* Noboru Harada

## Citing this work

If you'd like to cite this work, you may use the following. 

> Masahiro Yasuda, Yasunori Ohishi, Shoichiro Saito, Noboru Harada “Multi-view and Multi-modal Event Detection Utilizing Transformer-based Multi-sensor fusion,” in IEEE Int. Conf. Acoust. Speech Signal Process. (ICASSP), 2022.

## Link

Paper: [arXiv](hoge)
