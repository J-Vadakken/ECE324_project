'''
Download the SpiideoSynLoc and calibration-2023 datasets from SoccerNet and unzip the files.

Usage:
    python -m ECE324_Project.dataset_download
'''

from SoccerNet.Downloader import SoccerNetDownloader
mySoccerNetDownloader=SoccerNetDownloader(LocalDirectory="data/SoccerNet")
# mySoccerNetDownloader.downloadDataTask(task="SpiideoSynLoc", split=["train","valid","test","challenge"], version="fullhd")
mySoccerNetDownloader.downloadDataTask(task="SpiideoSynLoc", split=["train"], version="fullhd")

# mySoccerNetDownloader.downloadDataTask(task="calibration-2023", split=["train", "valid", "test"])


'''
To unzip the files:
    cd data/SoccerNet/SpiideoSynLoc
    for z in *.zip; do unzip $z; done

    cd data/SoccerNet/calibration-2023
    for z in *.zip; do unzip $z; done
'''