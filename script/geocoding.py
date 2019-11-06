import requests
from bs4 import BeautifulSoup
import time
from tqdm import tqdm
import pandas as pd
import numpy as np
import os

train = pd.read_csv("./input/train.csv")
test = pd.read_csv("./input/test.csv")
os.makedirs("./add_input", exist_ok=True)

train = train.rename(columns={'賃料': 'target', '契約期間': 'Contract', '間取り': 'Room',
                              '築年数': 'Passed', '駐車場': 'Parking', '室内設備': 'Facility',
                              '放送・通信': 'Internet', '周辺環境': 'Building', '建物構造': 'Material',
                              '面積': 'Area', 'キッチン': 'Kitchen', '所在地': 'Place',
                              'バス・トイレ': 'Bath', '所在階': 'Floor', 'アクセス': 'Access',
                              '方角': 'Angle'})
test = test.rename(columns={'契約期間': 'Contract', '間取り': 'Room', '築年数': 'Passed',
                            '駐車場': 'Parking', '室内設備': 'Facility', '放送・通信': 'Internet',
                            '周辺環境': 'Building', '建物構造': 'Material', '面積': 'Area',
                            'キッチン': 'Kitchen', '所在地': 'Place', 'バス・トイレ': 'Bath',
                            '所在階': 'Floor', 'アクセス': 'Access', '方角': 'Angle'})

# エラー回避用の補完
train['Place'].mask(train['Place'].str.endswith("丁目"), train['Place']+"1", inplace=True)
# test['Place'].mask(test['Place'].str.endswith("丁目"), test['Place']+"1", inplace=True)

URL = 'http://www.geocoding.jp/api/'


def coordinate(address):
    """
    addressに住所を指定すると緯度経度を返す。

    >>> coordinate('東京都文京区本郷7-3-1')
    ['35.712056', '139.762775']
    """
    payload = {'q': address}
    html = requests.get(URL, params=payload)
    soup = BeautifulSoup(html.content, "html.parser")
    if soup.find('error'):
        # raise ValueError(f"Invalid address submitted. {address}")
        print("geocode error")
        return [np.nan, np.nan]
    latitude = soup.find('lat').string
    longitude = soup.find('lng').string
    return [latitude, longitude]


def coordinates(addresses, interval=10, progress=True):
    """
    addressesに住所リストを指定すると、緯度経度リストを返す。

    >>> coordinates(['東京都文京区本郷7-3-1', '東京都文京区湯島３丁目３０−１'], progress=False)
    [['35.712056', '139.762775'], ['35.707771', '139.768205']]
    """
    coordinates = []
    for address in progress and tqdm(addresses) or addresses:
        coordinates.append(coordinate(address))
        time.sleep(interval)
    return coordinates

n_data = 1000
num = 24
for i in range(int(train.shape[0]/n_data)+1-num):
    # csvで保存
    train_geocode_list = coordinates(train["Place"].values[(i+num)*n_data:(i+1+num)*n_data])
    train_geocode_df = pd.DataFrame(train_geocode_list)
    train_geocode_df.to_csv("./add_input/train_geocode{}-{}.csv".format((i+num)*n_data, (i+1+num)*n_data))

    # test_geocode_list = coordinates(test["Place"].values)
    # test_geocode_df = pd.DataFrame(test_geocode_list)
    # test_geocode_df.to_csv("./add_input/test_geocode.csv")