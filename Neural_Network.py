import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
from streamlit_echarts import st_pyecharts
from pyecharts.charts import Line,Bar
from pyecharts import options as opts

import tensorflow.keras.layers as layers
from tensorflow import keras
import tensorflow as tf
from tensorflow.keras.models import load_model
import zipfile
import tempfile
import os
from sklearn.utils import resample
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
st.set_page_config(page_title="Neural Network")

def make_model(X_train, y_train,X_valid, y_valid,zip_f,n):
    model = keras.Sequential([
        layers.Dense(25, activation='relu', input_shape=[len(X_train.columns)]),
        # layers.Dense(64, activation='relu'),
        layers.Dense(1)
    ])
    
    optimizer = tf.keras.optimizers.RMSprop(0.001)
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', patience=10)
    model.compile(loss='mse',
                    optimizer=optimizer,
                    metrics=['mae', 'mse'])
    EPOCHS = 100
    model.fit(X_train, y_train, validation_data=(X_valid,y_valid),epochs=EPOCHS,verbose=5,callbacks=[early_stopping])
    y_pred = model.predict(X_valid)
    model_name = "model_"+str(n) + ".h5"
    model.save(model_name)
    zip_model(zip_f,model_name)
    return y_pred
    
def zip_model(zip_f,model_name):
    # ZIPファイルに追加
    zip_f.write(model_name, compress_type=zipfile.ZIP_DEFLATED)
    

@st.cache()
def read_data(train_file,test_file):
    train_df = pd.read_csv(train_file, encoding="shift-jis")
    test_df = pd.read_csv(test_file, encoding="shift-jis")
    return train_df, test_df

@st.cache()
def read_feature(feature_file,vote):
    feature_df = pd.read_csv(feature_file, encoding="utf-8-sig",index_col=0)
    if vote == "1,2":
        feature_df = feature_df["1"].dropna()
    else:
        feature_df = feature_df[vote].dropna()
    y_list = feature_df[feature_df.iloc[:] == "○"].index
    x_list = feature_df[feature_df.iloc[:] == "●"].index
    return x_list,y_list
    
def line_chart(x,y,y_name):
    b = (
            Line()
            .add_xaxis(x)
            .add_yaxis(y_name,
                 y,label_opts=opts.LabelOpts(is_show=False)
            )
            .set_global_opts(
                xaxis_opts=opts.AxisOpts(axislabel_opts=opts.LabelOpts(rotate=15,)),
                datazoom_opts=[opts.DataZoomOpts(), opts.DataZoomOpts(type_="inside")],
                title_opts=opts.TitleOpts(title="",),
            )
        )
    return b

def bar_chart(x,y,y_name):
    b = (
            Bar()
            .add_xaxis(x)
            .add_yaxis(y_name,
                 y,label_opts=opts.LabelOpts(is_show=False)
            )
            .set_global_opts(
                xaxis_opts=opts.AxisOpts(axislabel_opts=opts.LabelOpts(rotate=15,)),
                datazoom_opts=[opts.DataZoomOpts(), opts.DataZoomOpts(type_="inside")],
                title_opts=opts.TitleOpts(title="",),
            )
        )
    return b
# タイトル
st.title("Neural Network")
st.sidebar.title("Neural Network")

st.header("手順")
# ファイル入力
st.subheader("①学習用CSVファイル")
train_file = st.sidebar.file_uploader("学習用のCSVファイルを入力してください",type=["csv"])

if not train_file:
    st.warning("学習用のCSVファイルを入力してください")
    st.stop()
st.success("学習用CSVファイル入力完了")

st.subheader("②テスト用CSVファイル")
test_file = st.sidebar.file_uploader("テスト用のCSVファイルを入力してください",type=["csv"])

if not test_file:
    st.warning("テスト用のCSVファイルを入力してください")
    st.stop()
st.success("テスト用CSVファイル入力完了")

# 投票番号入力
st.subheader("③投票番号")
vote = st.sidebar.selectbox("投票番号を選択してください",("","1,2","3","4","5","6","7","8","9","10","11"))
if vote == "":
    st.warning("投票番号を入力してください")
    st.stop()
    
st.success(str(vote)+"を選択")
# 特徴量入力
st.subheader("④特徴量CSVデータ")
feature_file = st.sidebar.file_uploader("特徴量のCSVファイルを入力してください",type=["csv"])

if not feature_file:
    st.warning("特徴量のCSVファイルを入力してください")
    st.stop()
st.success("特徴量CSVファイル入力完了")
x_list,y_list = read_feature(feature_file,vote)

st.subheader("⑤データ読み込み")
with st.spinner("データを読み込んでいます"):
    train_df, test_df = read_data(train_file,test_file)
st.success("データ読み込み完了")

st.subheader("⑥データ前処理")
with st.spinner("前処理を実施中"):
    train_df = train_df[train_df["馬除外フラグ"] != 1]
    test_df = test_df[test_df["馬除外フラグ"] != 1]
st.success("データ前処理完了")

if vote == "1,2":
    train = train_df[(train_df["投票"]==1)|(train_df["投票"]==2)]
    test  = test_df[(test_df["投票"]==1)|(test_df["投票"]==2)]
    X_train = train[x_list]
    y_train = train[y_list]
    X_valid = test[x_list]
    y_valid = test[y_list]

else:
    train = train_df[(train_df["投票"]==int(vote))]
    test  = test_df[(test_df["投票"]==int(vote))]
    X_train = train[x_list]
    y_train = train[y_list]
    X_valid = test[x_list]
    y_valid = test[y_list]  

st.subheader("⑥モデル構築")
bagging_num = st.sidebar.number_input("バギングの数",value=200)

if not bagging_num:
    st.warning("バギングの数を入力してください")
    st.stop()

y_preds = pd.DataFrame()
  
with st.spinner("Neural Networkのモデルを構築しています"):
    with ThreadPoolExecutor(max_workers=100) as executor:
        with zipfile.ZipFile('./model.zip','w') as zip_f:
            futures = [executor.submit(make_model,X_train, y_train,X_valid, y_valid,zip_f,n) for n in range(bagging_num)]
            for future in as_completed(futures):
                y_pred = pd.DataFrame(future.result())
                y_preds = pd.concat([y_preds,y_pred],axis=1)

zip_f.close()
y_preds.columns=range(bagging_num)

st.success("モデル構築完了")
y_preds.columns=range(bagging_num)
all = y_preds.mean(axis='columns')
y_pred = pd.DataFrame(all.values)

st.header("データ分析")
with st.spinner("データ分析中"):
    y_pred.columns = ["pred"]
    y_pred = y_pred.reset_index(drop=True)
    test = test.reset_index(drop=True)
    out = pd.concat([test,y_pred],axis=1)
    prise_thre = st.sidebar.number_input("単勝配当予測閾値",value=100)
    out["購入馬"] = out.apply(lambda x: 1 if (x['pred'] >= prise_thre) else 0, axis=1)
    # 的中馬をマーキング
    out["hitmark"] = out.apply(lambda x: 1 if (x['着順'] == 1 and x['着順'] == x['購入馬']) else 0, axis=1)
    # 日ごと購入馬合計
    by_race = out[["レースキー","日付","hitmark","購入馬","単勝配当"]].groupby(["日付","レースキー"]).sum()
    # 的中率を計算
    acc_rate = out['hitmark'].sum() / len(by_race[by_race["購入馬"]>0]) * 100
    # 回収率を計算
    pay_back = out.loc[out['hitmark'] == 1, '単勝配当'].sum() / (out.loc[out['購入馬'] == 1, '購入馬'].sum()*100) * 100
    by_race["レース毎配当"] = by_race.apply(lambda x: x["単勝配当"] if (x['hitmark'] == 1 ) else 0, axis=1)
    by_race["レース毎損益"] = by_race.apply(lambda x: x["単勝配当"] - x["購入馬"]*100 if (x['hitmark'] == 1 ) else - x["購入馬"]*100, axis=1)
    by_race["累積損益"] = by_race["レース毎損益"].cumsum()
    col1, col2 = st.columns(2)
    with col1:
        st.metric("レース数",str(len(out['レースキー'].unique()))+"レース")
        st.metric("購入馬券数",str(len(out[out["購入馬"]>0]))+"枚")
        st.metric("的中馬券数",str(out['hitmark'].sum())+"枚")
    with col2:
        st.metric("的中率", str(int(acc_rate)) + "%")
        st.metric("回収率", str(int(pay_back)) + "%")
        st.metric("回収額", "¥"+str(by_race["累積損益"].tail(1).iloc[0]))
    
    by_race_data = by_race.reset_index()
    by_race_data["日付"] = pd.to_datetime(by_race_data['日付'], format='%Y%m%d')
    day= list(by_race_data["日付"].apply(str).str[:10])

    by_date = by_race_data.groupby(pd.Grouper(key="日付", freq="M")).sum()
    by_date = by_date.reset_index()
    day2= list(by_date["日付"].apply(str).str[:10])
   
    v1 = by_race_data["累積損益"]
    c = line_chart(day,v1,"累積損益")
    d = bar_chart(day2,by_date["購入馬"].values.tolist(),"購入馬券数")
    e = bar_chart(day2,by_date["hitmark"].values.tolist(),"的中馬券数")
    month_ticket = e.overlap(d)

    month_payback = (by_date["レース毎損益"] /(by_date["購入馬"]*100))*100
    month_payback_chart = bar_chart(day2,month_payback.values.tolist(),"月別回収率")
    st.subheader("累積損益")
    st_pyecharts(c)

    st.subheader("月別馬券数")
    st_pyecharts(month_ticket)
    st.subheader("月別回収率")
    st_pyecharts(month_payback_chart)

    

    with open("./model.zip", "rb") as fp:
        btn = st.download_button(
            label="Download model",
            data=fp,
            file_name="model.zip",
            mime="application/zip")
