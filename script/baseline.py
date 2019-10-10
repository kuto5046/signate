import pandas as pd
import pandas_profiling as pdp
import numpy as np
import seaborn as sns
import matplotlib
import requests
import time
from bs4 import BeautifulSoup
from tqdm import tqdm
import datetime
import matplotlib.pyplot as plt
import lightgbm as lgb
from sklearn.metrics import mean_squared_error, roc_auc_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# import japanize_matplotlib
# import geopy
# from geopy.geocoders import Nominatim
# from geopy.extra.rate_limiter import RateLimiter

date = datetime.datetime.today().strftime("%Y%m%d_%H%M%S")
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")

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


def feature_arrange(df, type):
    # 所在地
    place = df["Place"].str.replace("東京都", "").str.split("区")
    place = pd.DataFrame(place.str, index=["Place1", "Place2"]).T
    place = place["Place1"].str.replace("港", "0")
    place = place.str.replace("千代田", "1")
    place = place.str.replace("中央", "2")
    place = place.str.replace("渋谷", "3")
    place = place.str.replace("目黒", "4")

    place = place.str.replace("新宿", "5")
    place = place.str.replace("文京", "6")
    place = place.str.replace("台東", "7")
    place = place.str.replace("江東", "8")
    place = place.str.replace("品川", "9")

    place = place.str.replace("荒川", "10")
    place = place.str.replace("墨田", "11")
    place = place.str.replace("世田谷", "12")
    place = place.str.replace("豊島", "13")
    place = place.str.replace("大田", "14")

    place = place.str.replace("中野", "15")
    place = place.str.replace("北", "16")
    place = place.str.replace("杉並", "17")
    place = place.str.replace("練馬", "18")
    place = place.str.replace("板橋", "19")

    place = place.str.replace("江戸川", "20")
    place = place.str.replace("足立", "21")
    place = place.str.replace("葛飾", "22")
    place = pd.DataFrame(place).astype(int)
    place.columns = ["Place"]
    # place = pd.get_dummies(place["Place1"])
    # place = place.rename(columns={'世田谷': 'Setagaya', '中央': 'Cyuo', '中野': 'Nakano',
    #                               '北': 'Kita', '千代田': 'Tiyoda', '台東': 'Daito',
    #                               '品川': 'Sinagawa', '墨田': 'Sumida', '大田': 'Ota',
    #                               '文京': 'Bunkyo', '新宿': 'Sinzyuku', '杉並': 'Suginami',
    #                               '板橋': 'Itabashi', '江戸川': 'Edogawa', '江東': 'Eto',
    #                               '渋谷': 'Shibuya', '港': 'Minato', '目黒': 'Meguro', '練馬': 'Nerima',
    #                               '荒川': 'Arakawa', '葛飾': 'Katushika', '豊島': 'Toyoshima',
    #                               '足立': 'Adati'})
    # 間取り、方角, 建物構造
    # room = pd.get_dummies(df["Room"].str.replace("納戸", ""))
    # angle = pd.get_dummies(df["Angle"])
    # angle = angle.rename(columns={"北": "N", "北東": "NE", "北西": "NW", "南": "S",
    #                               "南東": "SE", "南西": "SW", "東": "E", "西": "W"})

    room = pd.DataFrame(df["Room"].str[0]).astype(int)
    room.columns = ["N_room"]
    # temp_room = df["Room"].str[1:].replace("R", "")
    # room["Living"] = temp_room.str.contains("L")
    # room["Dinnig"] = temp_room.str.contains("D")
    # room["Kitchen"] = temp_room.str.contains("K")
    # room["Sroom"] = temp_room.str.contains("S")
    # room = room * 1

    angle = df["Angle"].str.replace("北西", "1")
    angle = angle.str.replace("北東", "1")
    angle = angle.str.replace("南西", "3")
    angle = angle.str.replace("南東", "3")
    angle = angle.str.replace("北", "0")
    angle = angle.str.replace("西", "2")
    angle = angle.str.replace("東", "2")
    angle = angle.str.replace("南", "4")
    angle = pd.DataFrame(angle.fillna("2"), columns=["Angle"]).astype(int)
    
    # material = pd.get_dummies(df["Material"].str.replace("鉄筋ブロック", "ブロック"))
    # material = material.rename(columns={'ALC（軽量気泡コンクリート）': 'ALC', 'HPC（プレキャスト・コンクリート（重量鉄骨））': 'HPC',
    #                                     'PC（プレキャスト・コンクリート（鉄筋コンクリート））': 'PC', 'RC（鉄筋コンクリート）': 'RC',
    #                                     'SRC（鉄骨鉄筋コンクリート）': 'SRC', 'その他': 'Other', 'ブロック': 'Block',
    #                                     '木造': 'Wood', '軽量鉄骨': "LSteel", '鉄骨造': 'Steel'})
    material = df["Material"].str.replace("SRC（鉄骨鉄筋コンクリート）", "0")
    material = material.str.replace("RC（鉄筋コンクリート）", "0")
    material = material.str.replace("ALC（軽量気泡コンクリート）", "1")
    material = material.str.replace("PC（プレキャスト・コンクリート（鉄筋コンクリート））", "1")
    material = material.str.replace("軽量鉄骨", "2")
    material = material.str.replace("HPC（プレキャスト・コンクリート（重量鉄骨））", "2")
    material = material.str.replace("鉄骨造", "2")
    material = material.str.replace("木造", "3")
    material = material.str.replace("その他", "3")
    if type == "test":
        material = material.str.replace("鉄筋ブロック", "3")
    material = material.str.replace("ブロック", "4")
    material = material.str.replace("その他", "4")


    material = pd.DataFrame(material, columns=["Material"]).astype(int)


    # 築年数の前処理
    passed = df["Passed"].str.replace("新築", "0年").str.split("年", expand=True)
    passed = pd.DataFrame(passed[0]).astype(int)
    passed.columns = ["Passed"]
    # 面積
    area = df["Area"].str.replace('m2', '').astype(float).round()

    # 所在階
    floor = df["Floor"].str.replace('地下', '-').str.replace('階', '').str.split('／')
    floor = pd.DataFrame(floor.str, index=["LiveFloor", "MaxFloor"]).T

    # 最上階について
    max_floor = floor['MaxFloor'].fillna(floor['LiveFloor']).str.replace("建", "").str.split("（")
    max_floor = pd.DataFrame(max_floor.str, index=["MaxFloor", "UnderFloor"]).T
    max_floor = max_floor.fillna("1")  # テストデータの欠損値補間用

    # 所在階について
    live_floor = floor["LiveFloor"].str.replace("建", "").str.split("（")
    live_floor = pd.DataFrame(live_floor.str, index=["LiveFloor", "UnderFloor"]).T
    live_floor["LiveFloor"].mask(live_floor["LiveFloor"] == "", max_floor["MaxFloor"], inplace=True)
    live_floor = live_floor.fillna("1")  # テストデータの欠損値補間用

    max_floor = max_floor["MaxFloor"].astype(int)
    live_floor = live_floor["LiveFloor"].astype(int)
    floor = pd.concat([live_floor, max_floor], axis=1)
    floor["RatioFloor"] = ((live_floor / max_floor) * 100).round()

    # 浴槽とか
    bath = pd.DataFrame(df["Bath"].str.split("／", expand=True).nunique(axis=1), columns=["Bath"])
    kitchen = pd.DataFrame(df["Kitchen"].str.split("／", expand=True).nunique(axis=1), columns=["Kitchen"])
    facility = pd.DataFrame(df["Facility"].str.split("／", expand=True).nunique(axis=1), columns=["Facility"])
    internet = pd.DataFrame(df["Internet"].str.split("／", expand=True).nunique(axis=1), columns=["Internet"])

    if type == "train":
        target = df['target']
        new_df = pd.concat([target, place, room, passed, angle, area, floor, material,
                            bath, kitchen, facility, internet], axis=1)
    else:
        new_df = pd.concat([place, room, passed, angle, area, floor, material,
                            bath, kitchen, facility, internet], axis=1)
    return new_df


train = feature_arrange(train, type="train")
test = feature_arrange(test, type="test")

X = train.drop(['target'], axis=1)
y = train['target']
X_test = test

y_pred = np.zeros(X_test.shape[0], dtype='float32')
train_pred = np.zeros(X.shape[0], dtype='float32')

X_train, X_val, y_train, y_val = train_test_split(X, y, shuffle=True, random_state=0)
train_data = lgb.Dataset(X_train, y_train)
val_data = lgb.Dataset(X_val, y_val)
params = {
    'objective': 'regression',
    'metric': 'rmse',
    'learning_rate': 0.01,
    'max_depth': -1,
    # 'num_leaves': 255,
    # 'max_bin': 255,
    # 'colsample_bytree': 0.8,
    # 'subsample': 0.8,
    # 'nthread': -1,
    # 'bagging_freq': 1,
    'verbose': -1,
    'seed': 0,
}

model = lgb.train(params, train_data, valid_sets=[train_data, val_data],
                  num_boost_round=5000, early_stopping_rounds=200,
                  verbose_eval=200)

"""
# 線形回帰(標準化が必要)
from sklearn.linear_model import LinearRegression
model = LinearRegression()

model.fit(X_train, y_train)
"""


"""
# ランダムフォレスト回帰
params = {
    "n_estimators": 100,
    "random_state": 0,
    "max_depth": 7,
    }

model = RandomForestRegressor(**params)
model.fit(X_train, y_train)
"""

y_train_pred = model.predict(X_train)
y_val_pred = model.predict(X_val)
train_score = np.sqrt(mean_squared_error(y_train, y_train_pred))
val_score = np.sqrt(mean_squared_error(y_val, y_val_pred))

print('train_RMSE:', train_score)
print('val_RMSE:', val_score)

feature_importances = pd.DataFrame()
feature_importances['feature'] = X.columns
feature_importances['importance'] = model.feature_importance()
feature_importances = feature_importances.sort_values(by='importance', ascending=False)
plt.figure(figsize=(16, 16))
sns.barplot(data=feature_importances.head(50), x='importance', y='feature')
plt.show()

submit = pd.read_csv('../input/sample_submit.csv', names=('id', 'target'))
y_pred = model.predict(X_test)
submit['target'] = y_pred
submit.to_csv('../output/submit{}.csv'.format(date), header=False, index=False)
