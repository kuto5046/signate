import pandas as pd
import pandas_profiling as pdp
import numpy as np
import seaborn as sns
import matplotlib
import requests
import time
from bs4 import BeautifulSoup
from tqdm import tqdm
import matplotlib.pyplot as plt
import lightgbm as lgb
from sklearn.metrics import mean_squared_error, roc_auc_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
# import japanize_matplotlib
# import geopy
# from geopy.geocoders import Nominatim
# from geopy.extra.rate_limiter import RateLimiter


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
    place = pd.get_dummies(place["Place1"])
    place = place.rename(columns={'世田谷': 'Setagaya', '中央': 'Cyuo', '中野': 'Nakano',
                                  '北': 'Kita', '千代田': 'Tiyoda', '台東': 'Daito',
                                  '品川': 'Sinagawa', '墨田': 'Sumida', '大田': 'Ota',
                                  '文京': 'Bunkyo', '新宿': 'Sinzyuku', '杉並': 'Suginami',
                                  '板橋': 'Itabashi', '江戸川': 'Edogawa', '江東': 'Eto',
                                  '渋谷': 'Shibuya', '港': 'Minato', '目黒': 'Meguro', '練馬': 'Nerima',
                                  '荒川': 'Arakawa', '葛飾': 'Katushika', '豊島': 'Toyoshima',
                                  '足立': 'Adati'})
    # 間取り、方角, 建物構造
    # room = pd.get_dummies(df["Room"].str.replace("納戸", ""))
    angle = pd.get_dummies(df["Angle"])
    angle = angle.rename(columns={"北": "N", "北東": "NE", "北西": "NW", "南": "S",
                                  "南東": "SE", "南西": "SW", "東": "E", "西": "W" })
    material = pd.get_dummies(df["Material"].str.replace("鉄筋ブロック", "ブロック"))
    material = material.rename(columns={'ALC（軽量気泡コンクリート）': 'ALC', 'HPC（プレキャスト・コンクリート（重量鉄骨））': 'HPC',
                                        'PC（プレキャスト・コンクリート（鉄筋コンクリート））': 'PC', 'RC（鉄筋コンクリート）': 'RC',
                                        'SRC（鉄骨鉄筋コンクリート）': 'SRC', 'その他': 'Other','ブロック': 'Block',
                                        '木造': 'Wood', '軽量鉄骨': "LSteel", '鉄骨造': 'Steel'})
    # 築年数の前処理
    passed = df["Passed"].str.replace("新築", "0年").str.split("年", expand=True)
    passed = pd.DataFrame(passed[0]).astype(int)

    # 面積
    area = df["Area"].str.replace('m2', '').astype(float).astype(int)

    # 所在階
    floor = df["Floor"].str.replace('地下[1-9]{1,}', '').str.replace('階', '').str.replace('（）', '').str.replace('建', '').str.split('／')
    floor = pd.DataFrame(floor.str, index=["Floor", "MaxFloor"]).T
    floor["MaxFloor"] = floor['MaxFloor'].fillna(floor['Floor'])
    floor["MaxFloor"] = floor['MaxFloor'].fillna("4")  # テストデータの欠損値を埋める用
    max_floor = floor["MaxFloor"].astype(int)

    if type == "train":
        target = df['target']
        new_df = pd.concat([target, place, passed, angle, area, max_floor, material], axis=1)
    else:
        new_df = pd.concat([place, passed, angle, area, max_floor, material], axis=1)
    return new_df


train = feature_arrange(train, type="train")
test = feature_arrange(test, type="test")

X = train.drop(['target'], axis=1)
y = train['target']
X_test = test

y_pred = np.zeros(X_test.shape[0], dtype='float32')
train_pred = np.zeros(X.shape[0], dtype='float32')

cv_score = 0
X_train, X_valid, y_train, y_valid = train_test_split(X, y, random_state=0)
train_data = lgb.Dataset(X_train, y_train)
valid_data = lgb.Dataset(X_valid, y_valid)

params = {
    'objective': 'regression',
    'metric': 'rmse',
    'learning_rate': 0.01,
    'max_depth': -1,
    'num_leaves': 255,
    'max_bin': 255,
    'colsample_bytree': 0.8,
    'subsample': 0.8,
    'nthread': -1,
    'bagging_freq': 1,
    'verbose': -1,
    'seed': 1,
}

model = lgb.train(params, train_data, valid_sets=[train_data, valid_data],
                  num_boost_round=5000, early_stopping_rounds=200,
                  verbose_eval=200)

y_val_pred = model.predict(X_valid)
val_score = np.sqrt(mean_squared_error(y_valid, y_val_pred))
print('RMSE:', val_score)

y_pred += model.predict(X_test, num_iteration=model.best_iteration)

cv_score += val_score
print(cv_score)


feature_importances = pd.DataFrame()
feature_importances['feature'] = X.columns
feature_importances['importance'] = model.feature_importance()
feature_importances = feature_importances.sort_values(by='importance', ascending=False)
plt.figure(figsize=(16, 16))
sns.barplot(data=feature_importances.head(50), x='importance', y='feature')

submit = pd.read_csv('../input/sample_submit.csv', names=('id', 'target'))
submit['target'] = y_pred
submit.head(10)

submit.to_csv('submit_kernel_01.csv', header=False, index=False)