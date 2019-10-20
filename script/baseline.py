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
import os

print(os.getcwd())
# import japanize_matplotlib
# import geopy
# from geopy.geocoders import Nominatim
# from geopy.extra.rate_limiter import RateLimiter

date = datetime.datetime.today().strftime("%Y%m%d_%H%M%S")
train = pd.read_csv("./input/train.csv")
test = pd.read_csv("./input/test.csv")
submit = pd.read_csv('./input/sample_submit.csv', names=('id', 'target'))

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

# 外れ値除去&修正
train.drop(train.query("id==20428 or id == 20232").index, inplace=True)  # 築年数が異常に大きい
train.drop(train.query("id==20927").index, inplace=True)  # 面積あたりの価格が安すぎる（荒川区）
train.drop(train.query("id==7492").index, inplace=True)  # 面積あたりの価格が高すぎる(豊島区)
train.drop(train.query("id==5776").index, inplace=True)
# train.query("id==5776")["target"] /= 10  # 価格の桁が1つ多い(港区)


# lightGBMのcvメソッドから学習済みモデルを受け取れるように設定
class ModelExtractionCallback(object):
    """lightgbm.cv() から学習済みモデルを取り出すためのコールバックに使うクラス

    NOTE: 非公開クラス '_CVBooster' に依存しているため将来的に動かなく恐れがある
    """

    def __init__(self):
        self._model = None

    def __call__(self, env):
        # _CVBooster の参照を保持する
        self._model = env.model

    def _assert_called_cb(self):
        if self._model is None:
            # コールバックが呼ばれていないときは例外にする
            raise RuntimeError('callback has not called yet')

    @property
    def boosters_proxy(self):
        self._assert_called_cb()
        # Booster へのプロキシオブジェクトを返す
        return self._model

    @property
    def raw_boosters(self):
        self._assert_called_cb()
        # Booster のリストを返す
        return self._model.boosters

    @property
    def best_iteration(self):
        self._assert_called_cb()
        # Early stop したときの boosting round を返す
        return self._model.best_iteration


"""
特徴量の整理
    args: DataFrame
    return: DataFrame
"""

# 所在地
def place_feature(df):
    place = df["Place"].str.replace("東京都", "").str.split("区")
    place = pd.DataFrame(place.str, index=["Place", "Place2"]).T
    place.drop("Place2", axis=1, inplace=True)
    # place = place["Place"].str.replace("港", "0")
    # place = place.str.replace("千代田", "1")
    # place = place.str.replace("中央", "2")
    # place = place.str.replace("渋谷", "3")
    # place = place.str.replace("目黒", "4")

    # place = place.str.replace("新宿", "5")
    # place = place.str.replace("文京", "6")
    # place = place.str.replace("台東", "7")
    # place = place.str.replace("江東", "8")
    # place = place.str.replace("品川", "9")

    # place = place.str.replace("荒川", "10")
    # place = place.str.replace("墨田", "11")
    # place = place.str.replace("世田谷", "12")
    # place = place.str.replace("豊島", "13")
    # place = place.str.replace("大田", "14")

    # place = place.str.replace("中野", "15")
    # place = place.str.replace("北", "16")
    # place = place.str.replace("杉並", "17")
    # place = place.str.replace("練馬", "18")
    # place = place.str.replace("板橋", "19")

    # place = place.str.replace("江戸川", "20")
    # place = place.str.replace("足立", "21")
    # place = place.str.replace("葛飾", "22")
    # place = pd.DataFrame(place).astype(int)
    # place.columns = ["Place"]

    # frequency encording(区名の出現頻度を特徴量とする)
    freq = place["Place"].value_counts()
    place["Freq_place"] = place["Place"].map(freq)
    return place


# 間取り
def room_feature(df):
    room = pd.DataFrame(df["Room"].str[0]).astype(int)  # strの１文字目（部屋数）を取得
    room.columns = ["N_room"]
    temp_room = df["Room"].str[1:].replace("R", "")
    room["L_room"] = temp_room.str.contains("L")
    room["D_room"] = temp_room.str.contains("D")
    room["K_room"] = temp_room.str.contains("K")
    room["S_room"] = temp_room.str.contains("S")
    return room * 1


def floor_feature(df):
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
    return floor


# 方角
def angle_feature(df):
    angle = df["Angle"].str.replace("北西", "1")
    angle = angle.str.replace("北東", "1")
    angle = angle.str.replace("南西", "3")
    angle = angle.str.replace("南東", "3")
    angle = angle.str.replace("北", "0")
    angle = angle.str.replace("西", "2")
    angle = angle.str.replace("東", "2")
    angle = angle.str.replace("南", "4")
    angle = pd.DataFrame(angle.fillna("2"), columns=["Angle"]).astype(int)
    return angle


# def material_feature(df):
    # material = df["Material"].str.replace("鉄筋ブロック", "ブロック")
    # material = df["Material"].str.replace("SRC（鉄骨鉄筋コンクリート）", "0")
    # material = material.str.replace("RC（鉄筋コンクリート）", "0")
    # material = material.str.replace("ALC（軽量気泡コンクリート）", "1")
    # material = material.str.replace("PC（プレキャスト・コンクリート（鉄筋コンクリート））", "1")
    # material = material.str.replace("軽量鉄骨", "2")
    # material = material.str.replace("HPC（プレキャスト・コンクリート（重量鉄骨））", "2")
    # material = material.str.replace("鉄骨造", "2")
    # material = material.str.replace("木造", "3")
    # material = material.str.replace("その他", "3")

    # if type == "test":
    #     material = material.str.replace("鉄筋ブロック", "3")

    # material = material.str.replace("ブロック", "4")
    # material = material.str.replace("その他", "4")
    # material = pd.DataFrame(material, columns=["Material"]).astype(int)
    # return material


# 築年数
def passed_feature(df):
    passed = df["Passed"].str.replace("新築", "0年").str.split("年", expand=True)
    passed = pd.DataFrame(passed[0]).astype(int)
    passed.columns = ["Passed"]
    return passed


def feature_concat(df, data):
    place = place_feature(df)
    room = room_feature(df)
    passed = passed_feature(df)
    angle = angle_feature(df)
    area = df["Area"].str.replace('m2', '').astype(float).round()
    floor = floor_feature(df)
    material = df["Material"].str.replace("鉄筋ブロック", "ブロック")
    bath = pd.DataFrame(df["Bath"].str.split("／", expand=True).nunique(axis=1), columns=["Bath"])
    kitchen = pd.DataFrame(df["Kitchen"].str.split("／", expand=True).nunique(axis=1), columns=["Kitchen"])
    facility = pd.DataFrame(df["Facility"].str.split("／", expand=True).nunique(axis=1), columns=["Facility"])
    internet = pd.DataFrame(df["Internet"].str.split("／", expand=True).nunique(axis=1), columns=["Internet"])

    if data == "train":
        target = df['target']
        new_df = pd.concat([target, place, room, passed, angle, area, floor, material,
                            bath, kitchen, facility, internet], axis=1)
    else:
        new_df = pd.concat([place, room, passed, angle, area, floor, material,
                            bath, kitchen, facility, internet], axis=1)
    return new_df


def data_organize(train, test):
    train_df = feature_concat(train, data="train")
    test_df = feature_concat(test, data="test")

    # カテゴリ変数 for label encording
    le_columns = ["Place", "Material"]
    for c in le_columns:
        le = LabelEncoder()
        le.fit(train_df[c])
        train_df[c] = le.transform(train_df[c])
        test_df[c] = le.transform(test_df[c])

    X = train_df.drop(['target'], axis=1)
    y = train_df['target']
    X_test = test_df
    return X, y, X_test


def _feature_importance(model, X):
    feature_importances = pd.DataFrame()
    feature_importances['feature'] = X.columns
    feature_importances['importance'] = model.feature_importance()
    feature_importances = feature_importances.sort_values(by='importance', ascending=False)
    plt.figure(figsize=(8, 8))
    sns.barplot(data=feature_importances.head(50), x='importance', y='feature')
    plt.show()


def main():
    X, y, X_test = data_organize(train, test)
    # X_train, X_val, y_train, y_val = train_test_split(X, y, shuffle=True, random_state=0)
    # train_data = lgb.Dataset(X_train, y_train)
    # val_data = lgb.Dataset(X_val, y_val)
    train_data = lgb.Dataset(X, y)

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

    # 学習済みモデルを取り出すためのコールバックを用意する
    extraction_cb = ModelExtractionCallback()
    callbacks = [
        extraction_cb,
    ]

    lgb.cv(params, train_data, num_boost_round=5000, nfold=5, early_stopping_rounds=200, 
                   verbose_eval=200, callbacks=callbacks, eval_train_metric=True)

    # コールバックのオブジェクトから学習済みモデルを取り出す
    proxy = extraction_cb.boosters_proxy
    # boosters = extraction_cb.raw_boosters
    best_iteration = extraction_cb.best_iteration
    print(proxy)
    # _feature_importance(proxy, X)

    # 各モデルの推論結果を Averaging する場合
    y_pred_proba_list = proxy.predict(X_test, num_iteration=best_iteration)
    y_pred = np.array(y_pred_proba_list).mean(axis=0)
    # y_pred = np.zeros(X_test.shape[0], dtype='float32')
    # y_pred = np.argmax(y_pred_proba_avg, axis=1)

    # y_val_pred = model.predict(X_val)
    # train_score = np.sqrt(mean_squared_error(y_train, y_train_pred))
    # val_score = np.sqrt(mean_squared_error(y_val, y_val_pred))

   
    # y_pred = model.predict(X_test)
    submit['target'] = y_pred
    submit.to_csv('./output/submit{}.csv'.format(date), header=False, index=False)


if __name__ == "__main__":
    main()