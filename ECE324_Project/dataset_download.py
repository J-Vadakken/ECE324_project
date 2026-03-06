'''
Download the SpiideoSynLoc dataset from SoccerNet and unzip the files.

Usage:
    python -m ECE324_Project.dataset_download
'''

from SoccerNet.Downloader import SoccerNetDownloader
mySoccerNetDownloader=SoccerNetDownloader(LocalDirectory="data/SoccerNet")
mySoccerNetDownloader.downloadDataTask(task="SpiideoSynLoc", split=["train","valid","test","challenge"], version="fullhd")


'''
To unzip the files:
    cd data/SoccerNet/SpiideoSynLoc
    for z in *.zip; do unzip $z; done

'''