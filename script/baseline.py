import os
import time
import datetime
import pickle
import numpy as np
import pandas as pd
import seaborn as sns
import pandas_profiling as pdp
import matplotlib.pyplot as plt

import lightgbm as lgb
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, KFold

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
    
    #  ~丁目がつくか否か
    place["Place_tyome"] = place["Place2"].str.contains("丁目")

    #  ~町がつくか否か
    place["Place_machi"] = place["Place2"].str.contains("町")

    # frequency encording(区名の出現頻度を特徴量とする)
    freq = place["Place"].value_counts()
    place["Freq_place"] = place["Place"].map(freq)

    place.drop("Place2", axis=1, inplace=True)
    return place


# 間取り
def room_feature(df):
    room = pd.DataFrame(df["Room"].str[0]).astype(int)  # strの１文字目（部屋数）を取得
    room.columns = ["N_room"]
    room["Area_per_nroom"] = df["Area"].str.replace('m2', '').astype(float).round() / room["N_room"]
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

# アクセス
def access_feature(df):
    # 最寄りのアクセスの個数
    access = df["Access"].str.split("分", expand=True)
    access["N_access"] = access.count(axis=1)
    return access["N_access"]


# 方角
# def angle_feature(df):
#     angle = df["Angle"].str.replace("北西", "1")
#     angle = angle.str.replace("北東", "1")
#     angle = angle.str.replace("南西", "3")
#     angle = angle.str.replace("南東", "3")
#     angle = angle.str.replace("北", "0")
#     angle = angle.str.replace("西", "2")
#     angle = angle.str.replace("東", "2")
#     angle = angle.str.replace("南", "4")
#     angle = pd.DataFrame(angle.fillna("2"), columns=["Angle"]).astype(int)
#     return angle


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


# 近辺の建物
def building_feature(df):
    building = df["Building"].str.split("m", expand=True)
    building["N_building"] = building.count(axis=1)
    return building["N_building"]

# 面積
def area_feature(df, floor_df):
    area = df["Area"].str.replace('m2', '').astype(float).round()
    area = pd.DataFrame(area)
    area["Place"] = df["Place"].str.replace("東京都", "").str.split("区", expand=True)[0]
    area_mean = area.groupby("Place")["Area"].mean()
    area["Area_mean_place"] = area["Place"].map(area_mean)
    area["Area_mean_minus"] = area["Area"] - area["Area_mean_place"]
    area["Area_mean_division"] = area["Area"] / area["Area_mean_place"]
    area.drop("Place", axis=1, inplace=True)
    area["Volume"] = area["Area"] * floor_df["MaxFloor"]
    return area


def feature_concat(df, data):
    place = place_feature(df)
    access = access_feature(df)
    room = room_feature(df)
    passed = passed_feature(df)
    angle = df["Angle"].fillna("欠損値")  # 欠損値を１つのカテゴリとして処理
    floor = floor_feature(df)
    area = area_feature(df, floor)
    bath = pd.DataFrame(df["Bath"].str.split("／", expand=True).nunique(axis=1), columns=["Bath"])
    kitchen = pd.DataFrame(df["Kitchen"].str.split("／", expand=True).nunique(axis=1), columns=["Kitchen"])
    facility = pd.DataFrame(df["Facility"].str.split("／", expand=True).nunique(axis=1), columns=["Facility"])
    internet = pd.DataFrame(df["Internet"].str.split("／", expand=True).nunique(axis=1), columns=["Internet"])
    building = building_feature(df)
    material = df["Material"].str.replace("鉄筋ブロック", "ブロック")

    if data == "train":
        target = df['target']
        new_df = pd.concat([target, place, access, room, passed, angle, area, floor,
                            bath, kitchen, facility, internet, building, material], axis=1)
    else:
        new_df = pd.concat([place, access, room, passed, angle, area, floor,
                            bath, kitchen, facility, internet, building, material], axis=1)
    return new_df


def data_organize(train, test):
    train_df = feature_concat(train, data="train")
    test_df = feature_concat(test, data="test")

    # カテゴリ変数 for label encording
    le_columns = ["Place", "Angle", "Material"]
    for c in le_columns:
        le = LabelEncoder()
        le.fit(train_df[c])
        train_df[c] = le.transform(train_df[c])
        test_df[c] = le.transform(test_df[c])

    train_df["Tika"] = train_df["target"] / train_df["Area"]
    X = train_df.drop(['target'], axis=1)
    y = train_df['target']
    X_test = test_df
    return X, y, X_test


def _feature_importance(model, X):
    feature_importances = pd.DataFrame()
    feature_importances['feature'] = X.columns
    feature_importances['importance'] = model.feature_importance()
    feature_importances = feature_importances.sort_values(by='importance', ascending=False)
    plt.figure(figsize=(10, 10))
    sns.barplot(data=feature_importances.head(50), x='importance', y='feature')
    plt.show()


def target_encoding(X, y, tr_idx, va_idx):
    # 学習データからバリデーションデータを分ける
    tr_x, va_x = X.iloc[tr_idx].copy(), X.iloc[va_idx].copy()
    tr_y, va_y = y.iloc[tr_idx], y.iloc[va_idx]

    # trainデータを用いて,valデータをtarget encoding
    le_columns = ["Place", "Angle", "Material"]
    for c in le_columns:
        # 学習データ全体で各カテゴリにおけるtargetの平均を計算
        data_tmp1 = pd.DataFrame({c: tr_x[c], "target": tr_y})  # targetでtarget encoding
        data_tmp2 = pd.DataFrame({c: tr_x[c], "Tika": tr_x["Tika"]})  # tikaでtarget encoding
        target_mean = data_tmp1.groupby(c)["target"].mean()
        tika_mean = data_tmp2.groupby(c)["Tika"].mean()
        va_x.loc[:, c + "_te"] = va_x[c].map(target_mean)
        va_x.loc[:, c + "_tika_te"] = va_x[c].map(tika_mean)

        # trainデータの変換後の値を格納する配列を準備
        tmp1 = np.repeat(np.nan, tr_x.shape[0])
        tmp2 = np.repeat(np.nan, tr_x.shape[0])
        kf_encoding = KFold(n_splits=5, shuffle=True, random_state=0) 
        
        # trainデータを用いて,残りのtrainデータをtarget encoding
        for idx_1, idx_2 in kf_encoding.split(tr_x):
            # out-of-foldで各カテゴリにおける目的変数の平均を計算
            target_mean = data_tmp1.iloc[idx_1].groupby(c)["target"].mean()
            tika_mean = data_tmp2.iloc[idx_1].groupby(c)["Tika"].mean()
            # 変換後の値を一時配列に格納
            tmp1[idx_2] = tr_x[c].iloc[idx_2].map(target_mean)
            tmp2[idx_2] = tr_x[c].iloc[idx_2].map(tika_mean)
        tr_x.loc[:, c + "_te"] = tmp1
        tr_x.loc[:, c + "_tika_te"] = tmp2
    tr_x.drop("Tika", axis=1, inplace=True)
    va_x.drop("Tika", axis=1, inplace=True)
    return tr_x, tr_y, va_x, va_y


def main():
    X, y, X_test = data_organize(train, test)
    
    # 学習データからテストデータ用にtarget eoncoding
    le_columns = ["Place", "Angle", "Material"]
    for c in le_columns:
        data_tmp1 = pd.DataFrame({c: X[c], "target": y})  # targetでtarget encoding
        data_tmp2 = pd.DataFrame({c: X[c], "Tika": X["Tika"]})  # tikaでtarget encoding
        target_mean = data_tmp1.groupby(c)["target"].mean()
        tika_mean = data_tmp2.groupby(c)["Tika"].mean()
        X_test[c + "_te"] = X_test[c].map(target_mean)
        X_test[c + "_tika_te"] = X_test[c].map(tika_mean)

    kf = KFold(n_splits=5, shuffle=True, random_state=0)
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
        'verbose': 0,
        'seed': 0,
    }
    
    scores = []
    y_preds = []
    # 交差検証スタート
    for i, (tr_idx, va_idx) in enumerate(kf.split(X)):
        X_train, y_train, X_val, y_val = target_encoding(X, y, tr_idx, va_idx)

        train_data = lgb.Dataset(X_train, y_train)
        val_data = lgb.Dataset(X_val, y_val)

        # 学習済みモデルを取り出すためのコールバックを用意する
        # extraction_cb = ModelExtractionCallback()
        # callbacks = [extraction_cb,]
        model = lgb.train(params, train_set=train_data, num_boost_round=5000, valid_sets=val_data,
                          early_stopping_rounds=200, verbose_eval=200)

        # コールバックのオブジェクトから学習済みモデルを取り出す
        # proxy = extraction_cb.boosters_proxy
        # boosters = extraction_cb.raw_boosters
        # best_iteration = extraction_cb.best_iteration
        if i == 0:
            _feature_importance(model, X_train)
        
        # modelの保存
        # model_name = "model_fold{}.sav".format(i+1)
        # pickle.dump(model, open(model_name, "wb"))

        y_val_pred = model.predict(X_val)
        val_score = np.sqrt(mean_squared_error(y_val, y_val_pred))
        scores.append(val_score)
        print("valid_score {}: {}".format(i+1, val_score))

        y_preds.append(model.predict(X_test))


    score = np.mean(scores)
    print("mean_score: ", score)
    
    y_pred = np.mean(y_preds, axis=0)

    
    # # trainデータを用いて,testデータをtarget encoding
    # le_columns = ["Place", "Angle", "Material"]
    # for c in le_columns:
    #     # 学習データ全体で各カテゴリにおけるtargetの平均を計算
    #     data_tmp1 = pd.DataFrame({c: X_train[c], "target": y_train})
    #     data_tmp2 = pd.DataFrame({c: X_train[c], "Tika": tika_train})
    #     target_mean = data_tmp1.groupby(c)["target"].mean()
    #     tika_mean = data_tmp2.groupby(c)["Tika"].mean()
    #     X_test.loc[c + "_te"] = X_test[c].map(target_mean)
    #     X_test.loc[c + "_tika_te"] = X_test[c].map(tika_mean)
    

    submit['target'] = y_pred
    submit.to_csv('./output/submit{}.csv'.format(date), header=False, index=False)
    # print("train: ", X_train.columns)
    # print("test: ", X_test.columns)
if __name__ == "__main__":
    main()