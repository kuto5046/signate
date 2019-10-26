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


"""
特徴量の整理
    args: DataFrame
    return: DataFrame
"""
def feature(df):
    # -------
    # 所在地 
    # -------
    place = df["Place"].str.replace("東京都", "").str.split("区")
    place = pd.DataFrame(place.str, index=["Place", "Place2"]).T
    df["Place"] = place["Place"]  # 区をカテゴリとして特徴量
    df["Place_tyome"] = place["Place2"].str.contains("丁目") * 1  #  ~丁目がつくか否か
    df["Place_machi"] = place["Place2"].str.contains("町") * 1  #  ~町がつくか否か

    # frequency encording(区名の出現頻度を特徴量とする)
    freq = df["Place"].value_counts()
    df["Freq_place"] = place["Place"].map(freq)

    # -----
    # 面積
    # -----
    df["Area"] = df["Area"].str.replace('m2', '').astype(float).round()
    area_mean = df.groupby("Place")["Area"].mean()
    df["Area_mean_place"] = df["Place"].map(area_mean)
    df["Area_mean_minus"] = df["Area"] - df["Area_mean_place"]
    df["Area_mean_division"] = df["Area"] / df["Area_mean_place"]
    # 区ごとの面積の平均を特徴量に追加
    area_mean = df.groupby("Place")["Area"].mean()
    df["Area_mean_per_ward"] = train["Place"].map(area_mean)
    
    
    # ------ 
    # 間取り
    # ------
    df["N_room"] = df["Room"].str[0].astype(int)  # strの１文字目（部屋数）を取得
    df["Area_per_nroom"] = df["Area"] / df["N_room"]
    temp_room = df["Room"].str[1:].replace("R", "")
    df["L_room"] = temp_room.str.contains("L") * 1
    df["D_room"] = temp_room.str.contains("D") * 1
    df["K_room"] = temp_room.str.contains("K") * 1
    df["S_room"] = temp_room.str.contains("S") * 1
    df["N_room2"] = df["N_room"] + df["L_room"] + df["D_room"] + df["K_room"] + df["S_room"]

    
    # ------
    # 所在階
    # ------
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

    df["MaxFloor"] = max_floor["MaxFloor"].astype(int)
    df["LiveFloor"] = live_floor["LiveFloor"].astype(int)
    df["RatioFloor"] = ((df["MaxFloor"] / df["LiveFloor"]) * 100).round()
    df["Volume"] = df["Area"] * df["MaxFloor"]
    
    # ----
    # 方角
    # ----
    df["Angle"] = df["Angle"].fillna("欠損値")  # 欠損値を１つのカテゴリとして処理

    # -------
    # アクセス
    # -------
    # 最寄りのアクセスの個数
    access = df["Access"].str.split("分", expand=True)
    df["N_access"] = access.count(axis=1)
    
    # アクセスの最初の項を整理して特徴量に 
    one_access = access[0]
    one_access = one_access.str.replace("エクスプレス", "エクスプレス線")
    one_access = one_access.str.replace("ライン", "ライン線")
    one_access = one_access.str.replace("ライナー", "ライナー線")
    one_access = one_access.str.replace("ゆりかもめ", "ゆりかもめ線")
    one_access = one_access.str.split("線", expand=True)
    # one_access[1] = one_access[1].str.split("駅", expand=True)[0].str.replace("\t", "")
    df["First_line"] = one_access[0]

    # -----
    # 築年数
    # -----
    passed = df["Passed"].str.replace("新築", "0年").str.split("年", expand=True)
    df["Passed"] = passed[0].astype(int)
    passed_mean = df.groupby("Place")["Passed"].mean()
    df["Passed_mean_per_ward"] = df["Place"].map(passed_mean)


    # ---------
    # 近辺の建物
    # ---------
    building = df["Building"].str.split("m", expand=True)
    df["N_building"] = building.count(axis=1)

    # ----------
    # バス・トイレ
    # ----------
    temp = df["Bath"].fillna("／")
    df["N_bath"] = temp.str.split("／", expand=True).nunique(axis=1)
    df["A_bath"] = temp.str.contains("専用バス") * 1
    df["A_toilet"] = temp.str.contains("専用トイレ") * 1
    df["Separate_bath"] = temp.str.contains("バス・トイレ別") * 1
    df["Shower"] = temp.str.contains("シャワー") * 1
    df["A_senmendai"] = temp.str.contains("洗面台独立") * 1
    df["Co_toilet"] = temp.str.contains("共同トイレ") * 1
    df["Non_bath"] = temp.str.contains("バスなし") * 1
    df["Non_toilet"] = temp.str.contains("トイレなし") * 1
    df["Co_bath"] = temp.str.contains("共同バス") * 1
    df["Dryer"] = temp.str.contains("浴室乾燥機") * 1
    df["Reheater"] = temp.str.contains("追焚機能") * 1
    df["Hot_toilet"] = temp.str.contains("温水洗浄便座") * 1
    df["Dressing_room"] = temp.str.contains("脱衣所") * 1


    # --------
    # キッチン
    # --------
    temp = df["Kitchen"].fillna("／")
    df["N_kitchen"] = temp.str.split("／", expand=True).nunique(axis=1)
    df["Gas_stove"] = temp.str.contains("ガスコンロ") * 1
    df["Hot_water_supply"] = temp.str.contains("給湯") * 1
    df["Stove2"] = temp.str.contains("コンロ2口") * 1
    df["IH_stove"] = temp.str.contains("IHコンロ") * 1
    df["Stove3"] = temp.str.contains("コンロ3口") * 1
    df["Non_Stove2"] = temp.str.contains("コンロ設置可（コンロ2口") * 1
    df["System_kitchen"] = temp.str.contains("システムキッチン") * 1
    df["Ele_stove"] = temp.str.contains("電気コンロ") * 1
    df["Stove1"] = temp.str.contains("コンロ1口") * 1
    df["Non_stove_nan"] = temp.str.contains("コンロ設置可（口数不明") * 1
    df["Non_stove3"] = temp.str.contains("コンロ設置可（コンロ3口") * 1
    df["Non_stove1"] = temp.str.contains("コンロ設置可（コンロ1口）") * 1
    df["Refrigerator"] = temp.str.contains("冷蔵庫あり") * 1
    df["Counter_Kitchen"] = temp.str.contains("カウンターキッチン") * 1
    df["A_kitchen"] = temp.str.contains("独立キッチン") * 1
    df["Non_stove4"] = temp.str.contains("コンロ設置可（コンロ4口以上）") * 1
    df["Stove4"] = temp.str.contains("コンロ4口以上") * 1
    df["L_kitchen"] = temp.str.contains("L字キッチン") * 1

    # -----
    # 設備
    # -----
    df["Facility"] = pd.DataFrame(df["Facility"].str.split("／", expand=True).nunique(axis=1), columns=["Facility"])
    

    # ------------
    # インターネット
    # ------------
    temp = df["Internet"].fillna("／")
    df["N_internet"] = temp.str.split("／", expand=True).nunique(axis=1)
    df["An_Internet"] = temp.str.contains("インターネット対応") * 1
    df["Optical_fiber"] = temp.str.contains("光ファイバー") * 1
    df["Catv"] = temp.str.contains("CATV") * 1
    df["Hspeed_internet"] = temp.str.contains("高速インターネット") * 1
    df["CS"] = temp.str.contains("CSアンテナ") * 1
    df["BS"] = temp.str.contains("BSアンテナ") * 1
    df["Free_internet"] = temp.str.contains("インターネット使用料無料") * 1
    df["Cable"] = temp.str.contains("有線放送") * 1

    # --------
    # 建築材料
    # --------
    df["Material"] = df["Material"].str.replace("鉄筋ブロック", "ブロック")

    # 不要な列の削除
    df.drop(["Room", "Floor", "Facility", "Internet", "Access", "Building", "Bath", "Kitchen", "Parking", "Contract"], axis=1, inplace=True)
    return df
    
def label_encoding(train, test):
    train_df = feature(train)
    test_df = feature(test)
    test_df["First_line"] = test_df["First_line"].str.replace("中央本", "中央")
    test_df["First_line"] = test_df["First_line"].str.replace("埼玉高速鉄道", "南北")
    # ----------------
    # label encording
    # ----------------
    le_columns = ["Place", "Angle", "Material", "First_line"]

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
    le_columns = ["Place", "Angle", "Material", "First_line"]
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
    X, y, X_test = label_encoding(train, test)
    
    # 学習データからテストデータ用にtarget eoncoding
    le_columns = ["Place", "Angle", "Material", "First_line"]
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
        'num_leaves': 255,
        'max_bin': 255,
        'colsample_bytree': 0.8,
        'subsample': 0.8,
        'nthread': -1,
        'bagging_freq': 1,
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

        model = lgb.train(params, train_set=train_data, num_boost_round=5000, valid_sets=val_data,
                          early_stopping_rounds=200, verbose_eval=200)

        if i == 0:
            _feature_importance(model, X_train)


        y_val_pred = model.predict(X_val)
        val_score = np.sqrt(mean_squared_error(y_val, y_val_pred))
        scores.append(val_score)
        print("valid_score {}: {}".format(i+1, val_score))

        y_preds.append(model.predict(X_test))


    score = np.mean(scores)
    print("mean_score: ", score)
    
    y_pred = np.mean(y_preds, axis=0)

    submit['target'] = y_pred
    # submit.to_csv('./output/submit{}.csv'.format(date), header=False, index=False)
    # print("train: ", X_train.columns)
    # print("test: ", X_test.columns)
if __name__ == "__main__":
    main()