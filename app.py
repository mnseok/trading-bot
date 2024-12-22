from flask import Flask, render_template
import sqlite3
import pandas as pd
import subprocess

app = Flask(__name__)

# SQLite 데이터베이스 파일 경로
DB_PATH = "trading_records.db"

def fetch_trade_records():
    """SQLite에서 거래 기록을 가져옴"""
    try:
        conn = sqlite3.connect(DB_PATH)
        query = "SELECT * FROM trade_records ORDER BY timestamp DESC"
        df = pd.read_sql_query(query, conn)
        conn.close()
        return df
    except Exception as e:
        print(f"Error fetching trade records: {e}")
        return pd.DataFrame()

from datetime import datetime

@app.route("/")
def home():
    """홈페이지: 거래 기록 테이블 표시"""
    records = fetch_trade_records()
    if records.empty:
        message = "No trade records found."
        return render_template("index.html", records=[], columns=[], message=message)

    # 데이터프레임을 리스트로 변환 (행 및 열 인덱스 포함)
    records_with_index = []
    for row_index, row in enumerate(records.values):
        row_with_index = {"row_index": row_index, "cells": []}
        for cell_index, cell in enumerate(row):
            # timestamp 포맷팅
            if cell_index == 1:  # timestamp 열
                cell = datetime.strptime(cell, "%Y-%m-%dT%H:%M:%S.%f").strftime(
                    "%Y-%m-%d %H:%M:%S"
                )
            row_with_index["cells"].append({"cell_index": cell_index, "value": cell})
        records_with_index.append(row_with_index)

    return render_template("index.html", records=records_with_index, columns=records.columns)

def start_trading_bot():
    """Ensure the trading bot script is running."""
    try:
        # Replace 'trading_bot.py' with the actual filename of your bot script
        subprocess.Popen(["python", "mvp.py"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print("Trading bot script started.")
    except Exception as e:
        print(f"Error starting trading bot: {e}")

if __name__ == "__main__":
    # 거래 봇 실행
    # start_trading_bot()
    app.run(debug=True)
