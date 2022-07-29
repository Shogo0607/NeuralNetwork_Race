import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
from streamlit_echarts import st_pyecharts
from pyecharts.charts import Line,Bar
from pyecharts import options as opts
import zipfile
import tempfile
import os
import tensorflow.keras.layers as layers
from tensorflow import keras
import tensorflow as tf
from tensorflow.keras.models import load_model
from concurrent.futures import ThreadPoolExecutor, as_completed

st.set_page_config(page_title="Neural Network")
   
@st.cache()
def read_data(predict_file):
    predict_df = pd.read_csv(predict_file, encoding="shift-jis")
    return predict_df

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

@st.cache()
def read_model(stream): 
    model_list = list()
    myzipfile = zipfile.ZipFile(stream)
    with tempfile.TemporaryDirectory() as tmp_dir:
        myzipfile.extractall(tmp_dir)
        root_folder = myzipfile.namelist()
        for folder in root_folder:
            model_dir = os.path.join(tmp_dir, folder)
            model = load_model(model_dir)
            model_list.append(model)
    return model_list


def prediction(model,X_predict):
    y_pred = model.predict(X_predict)
    y_pred = pd.DataFrame(y_pred)
    return y_pred

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
st.subheader("①予測用CSVファイル")
predict_file = st.sidebar.file_uploader("予測用のCSVファイルを入力してください",type=["csv"])

if not predict_file:
    st.warning("予測用のCSVファイルを入力してください")
    st.stop()
st.success("予測用CSVファイル入力完了")

st.subheader("②学習済みモデル")
stream = st.sidebar.file_uploader('Model file (.zip)', type='zip')
if stream is None:
    st.warning("学習済みモデルを入力してください")
    st.stop()

model_list = read_model(stream)

st.success("学習済みモデル(.zip)入力完了")

# 投票番号入力
st.subheader("③投票番号")
vote = st.sidebar.selectbox("投票番号を選択してください",("","1,2","3","4","5","6","7","8","9","10","11"))
if not vote:
    st.warning("投票番号を選択してください")
    st.stop()

st.success("投票番号"+vote+"を選択")

# 投票番号入力
st.subheader("④特徴量CSVデータ")
feature_file = st.sidebar.file_uploader("特徴量のCSVファイルを入力してください",type=["csv"])

if not feature_file:
    st.warning("特徴量のCSVファイルを入力してください")
    st.stop()
st.success("特徴量CSVファイル入力完了")
x_list,y_list = read_feature(feature_file,vote)
st.subheader("⑤データ読み込み")
with st.spinner("データを読み込んでいます"):
    predict_df = read_data(predict_file)
st.success("データ読み込み完了")

st.subheader("⑥データ前処理")
with st.spinner("前処理を実施中"):
    predict_df = predict_df[predict_df["馬除外フラグ"] != 1]
st.success("データ前処理完了")

if vote == "1,2":
    predict = predict_df[(predict_df["投票"]==1)|(predict_df["投票"]==2)]
    X_predict = predict[x_list]
    y_predict = predict[y_list]

else:
    predict = predict_df[(predict_df["投票"]==int(vote))]
    X_predict = predict[x_list]
    y_predict = predict[y_list]

st.header("データ分析")
y_preds = pd.DataFrame()
with st.spinner("データ分析中"):
    for model in model_list:
        y_pred = prediction(model,X_predict)
        y_preds = pd.concat([y_preds,y_pred],axis=1)

    y_preds.columns=range(len(model_list))
    all = y_preds.mean(axis='columns')
    y_pred = pd.DataFrame(all.values)
    y_pred.columns = ["pred"]
    y_pred = pd.DataFrame(y_pred)
    y_pred.columns = ["pred"]
    y_pred = y_pred.reset_index(drop=True)
    predict = predict.reset_index(drop=True)
    out = pd.concat([predict,y_pred],axis=1)
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