import os
import time
import json
from datetime import datetime, timedelta

import pandas as pd
import requests
from dotenv import load_dotenv
import pyupbit

# ta에서 제공하는 지표
from ta.utils import dropna
from ta.trend import EMAIndicator, MACD, ADXIndicator
from ta.momentum import RSIIndicator, StochasticOscillator, WilliamsRIndicator
from ta.volatility import BollingerBands, KeltnerChannel, AverageTrueRange
from ta.volume import OnBalanceVolumeIndicator, VolumeWeightedAveragePrice

# Gemini + PIL
import google.generativeai as genai
from PIL import Image

# 차트 스크린샷 함수 임포트
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

# DB 저장용
import sqlite3

# 유튜브 자막 API
from youtube_transcript_api import YouTubeTranscriptApi

import os
import time
import json
import logging
from datetime import datetime, timedelta
import sys

# 기존 코드 상단에 추가: logging 설정
def setup_logging(log_file="trading_bot.log"):
    """Setup logging to log all outputs to a file."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s]: %(message)s",
        handlers=[
            logging.FileHandler(log_file, mode="a"),  # Append mode
        ]
    )

    # Redirect print to logging
    class LoggingWrapper:
        def write(self, message):
            if message.strip():
                logging.info(message.strip())
        def flush(self):  # Dummy flush to avoid AttributeError
            pass

    sys.stdout = LoggingWrapper()  # Redirect print
    sys.stderr = LoggingWrapper()  # Redirect errors (optional)
    
# pd.Timestamp -> isoformat 변환용
def convert_timestamps(data):
    if isinstance(data, dict):
        return {convert_timestamps(k): convert_timestamps(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [convert_timestamps(item) for item in data]
    elif isinstance(data, pd.Timestamp):
        return data.isoformat()
    else:
        return data


def take_fullpage_screenshot(url: str, screenshot_name: str = "full_screenshot.png"):
    """Upbit 차트 웹페이지 전체 스크린샷을 찍는 함수"""
    chrome_driver_path = "/opt/homebrew/bin/chromedriver"  # 환경에 맞게 수정
    service = Service(executable_path=chrome_driver_path)

    options = webdriver.ChromeOptions()
    options.add_argument("--headless")

    driver = webdriver.Chrome(service=service, options=options)
    wait = WebDriverWait(driver, 10)

    try:
        driver.get(url)
        time.sleep(1)

        # 차트 시간 단위를 1시간봉으로
        menu_button = driver.find_element(By.XPATH, '/html/body/div[1]/div[2]/div[3]/span/div/div/div[1]/div/div/cq-menu[1]')
        menu_button.click()
        time.sleep(0.5)

        one_hour_button = wait.until(EC.presence_of_element_located(
            (By.XPATH, '//cq-item[@stxtap="Layout.setPeriodicity(1,60,\'minute\')"]')
        ))
        one_hour_button.click()

        # Bollinger Band 지표 추가
        indicator_button = driver.find_element(By.XPATH, '/html/body/div[1]/div[2]/div[3]/span/div/div/div[1]/div/div/cq-menu[3]')
        indicator_button.click()

        bollinger_band_button = wait.until(EC.presence_of_element_located((
            By.CSS_SELECTOR,
            "#fullChartiq > div > div > div.ciq-nav > div > div > cq-menu.ciq-menu.ciq-studies.collapse.stxMenuActive > cq-menu-dropdown > cq-scroll > cq-studies > cq-studies-content > cq-item:nth-child(15)"
        )))
        bollinger_band_button.click()

        driver.fullscreen_window()
        time.sleep(1)

        driver.save_screenshot(screenshot_name)
        print(f"[INFO] Full-page screenshot saved as {screenshot_name}")

    except Exception as e:
        print("에러가 발생했습니다:", e)
    finally:
        driver.quit()
        
class BitcoinTrader:
    def __init__(self):
        # .env 파일 불러오기
        # setup_logging()
        
        load_dotenv()
        self.access = os.getenv("UPBIT_ACCESS_KEY")
        self.secret = os.getenv("UPBIT_SECRET_KEY")
        self.serpapi_key = os.getenv("SERPAPI_KEY")

        # Upbit 객체
        self.upbit = pyupbit.Upbit(self.access, self.secret)

        # DB 연결
        self.db_connection = sqlite3.connect("trading_records.db")
        self.db_cursor = self.db_connection.cursor()
        self.create_table()

        # 트레이딩할 티커
        self.ticker = "KRW-BTC"

        # (1) 유튜브 자막 불러오기
        video_id = "3XbtEX3jUv4"
        self.youtube_text = self.get_youtube_transcript(video_id)  # 클래스 멤버에 저장

        # (2) 이전 반성문 가져오기 → 초기 시스템 메시지에 활용
        reflections_texts = self.load_all_reflections(limit=5)
        joined_reflections = "\n\n".join(reflections_texts) if reflections_texts else ""

        # Gemini API 설정
        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

        # (A) 트레이딩 의사결정용 Model
        trading_system_message = f"""
You are an expert in Bitcoin trading, specializing in '워뇨띠'님의 매매법.
You consider price trends, volume patterns, order book imbalances, 
market volatility, the Fear & Greed Index, relevant news, and the video transcript.

Here is a transcript from the 워뇨띠's trading method video:
{self.youtube_text}

Here are previous trade reflections:
{joined_reflections}

Your job: Provide a buy/sell/hold decision & rationale.
        """.strip()

        trading_gen_config = {
            "temperature": 1,
            "top_p": 0.9,
            "top_k": 40,
            "max_output_tokens": 8192,
            "response_mime_type": "application/json",
        }

        self.model_trading = genai.GenerativeModel(
            # model_name="gemini-1.5-flash",
            model_name="gemini-2.0-flash-exp",
            system_instruction=trading_system_message,
            generation_config=trading_gen_config
        )

        # (B) 반성문(Reflection) 전용 Model
        reflection_system_message = """
You are a reflection AI that provides constructive feedback on trading decisions. 
Your style: objective, concise, and offering improvements for the future.
""".strip()

        reflection_gen_config = {
            "temperature": 0.7,
            "top_p": 0.9,
            "top_k": 40,
            "max_output_tokens": 1024,
            "response_mime_type": "application/json",
        }

        self.model_reflection = genai.GenerativeModel(
            model_name="gemini-1.5-flash",
            system_instruction=reflection_system_message,
            generation_config=reflection_gen_config
        )

    def create_table(self):
        """매매 기록 테이블 생성 (reflection, key_improvement_points, confidence_level 컬럼 추가)"""
        self.db_cursor.execute("""
        CREATE TABLE IF NOT EXISTS trade_records (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            action TEXT NOT NULL,
            trade_percent INTEGER NOT NULL,
            trade_amount REAL NOT NULL,
            reason TEXT,
            balance_after REAL NOT NULL,
            reflection TEXT,
            key_improvement_points TEXT,
            confidence_level REAL,
            coin_ticker TEXT NOT NULL
        );
        """)
        self.db_connection.commit()

    def get_youtube_transcript(self, video_id: str) -> str:
        """YouTubeTranscriptApi를 이용해서 자막 불러오기"""
        try:
            transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=['ko'])
            combined_text = " ".join(item['text'] for item in transcript)
            formatted_text = " ".join(combined_text.split())
            return formatted_text
        except Exception as e:
            print(f"Error fetching YouTube transcript: {e}")
            return ""

    def load_all_reflections(self, limit=5, coin_ticker=None):
        """가장 최근 N개 reflection, key_improvement_points, confidence_level을 가져와서 리스트로 반환"""
        try:
            if coin_ticker is None:
                coin_ticker = self.ticker

            self.db_cursor.execute("""
                SELECT reflection, key_improvement_points, confidence_level
                FROM trade_records
                WHERE reflection IS NOT NULL AND reflection != '' AND coin_ticker = ?
                ORDER BY timestamp DESC
                LIMIT ?
            """, (coin_ticker, limit))
            rows = self.db_cursor.fetchall()
            
            # rows는 [(reflection, kip, conf), (reflection, kip, conf), ...] 형태
            # 이 데이터를 문자열 형태로 합칠지, dict 형태로 반환할지 결정
            
            # (A) 문자열 형태로 바로 구성
            combined_list = []
            for reflection, kip, conf in rows:
                # 값이 None인 경우 빈 문자열로 처리
                reflection_str = reflection if reflection else "N/A"
                kip_str = kip if kip else "N/A"
                conf_str = conf if conf else "N/A"

                # 보기 좋게 합친다
                segment = (
                    f"Reflection: {reflection_str}\n"
                    f"Key Improvement Points: {kip_str}\n"
                    f"Confidence Level: {conf_str}"
                )
                combined_list.append(segment)

            return combined_list

        except Exception as e:
            print(f"Error loading reflections: {e}")
            return []

    def log_trade(self, action, trade_percent, trade_amount, reason, balance_after):
        """매매 기록을 데이터베이스에 저장"""
        timestamp = datetime.now().isoformat()
        self.db_cursor.execute("""
        INSERT INTO trade_records (timestamp, action, trade_percent, trade_amount, reason, balance_after, coin_ticker)
        VALUES (?, ?, ?, ?, ?, ?, ?);
        """, (timestamp, action, trade_percent, trade_amount, reason, balance_after, self.ticker))
        self.db_connection.commit()
        print(f"[DB] 매매 기록 저장 완료: {action} {trade_amount} at {timestamp}")

    def store_reflection_for_trade(self, trade_id, reflection_data):
        """
        반성문(Reflection), 개선점, 자신도(Confidence) DB 업데이트
        reflection_data 예시:
        {
            "reflection": "내용...",
            "key_improvement_points": "내용...",
            "confidence_level": "0.8"
        }
        """
        try:
            reflection_text = reflection_data.get("reflection", "")
            key_improvement_points = reflection_data.get("key_improvement_points", "")
            confidence_level = reflection_data.get("confidence_level", "")

            self.db_cursor.execute("""
                UPDATE trade_records
                SET reflection = ?,
                    key_improvement_points = ?,
                    confidence_level = ?
                WHERE id = ?;
            """, (reflection_text, key_improvement_points, confidence_level, trade_id))
            self.db_connection.commit()
            print(f"[DB] Reflection stored for trade_id={trade_id}")
        except Exception as e:
            print(f"Error storing reflection: {e}")

    def reflect_on_trades_and_market(self, recent_trades, analysis):
        """
        최근 매매내역 + 시장데이터(analysis)를 AI에게 전달해서 '반성문'을 받아 반환
        """
        if not recent_trades:
            return None

        print(recent_trades[0])
        trades_data = []
        for trade in recent_trades:
            # (id, timestamp, action, trade_percent, trade_amount, reason, balance_after, reflection, key_improvement_points, confidence_level)
            t_id, ts, action, t_percent, t_amount, reason, bal_after, refl, kip, conf, _ = trade
            trades_data.append({
                "id": t_id,
                "timestamp": ts,
                "action": action,
                "trade_percent": t_percent,
                "trade_amount": t_amount,
                "reason": reason,
                "balance_after": bal_after,
                "existing_reflection": refl,
                "key_improvement_points": kip,
                "confidence_level": conf
            })

        converted_analysis = convert_timestamps(analysis)
        reflection_prompt = f"""
Below are recent trades and current market analysis data.
Please provide a reflection on how effective the trades were, 
what went well or poorly, and how to improve next time.
Return a short but informative reflection in JSON format, e.g.:

{{
  "reflection": "I lost money on this trade because of ... ... Next time I will definitely ...",
  "key_improvement_points": "...",
  "confidence_level": "0.8"
}}

**Recent Trades**:
{json.dumps(trades_data, indent=2)}

**Current Market Analysis**:
{json.dumps(converted_analysis, indent=2)}
""".strip()

        try:
            response = self.model_reflection.generate_content([reflection_prompt])
            if hasattr(response.candidates[0].content, "parts"):
                raw_parts = response.candidates[0].content.parts
                raw_text = "".join([p.text for p in raw_parts])
            else:
                raw_text = response.candidates[0].content

            clean_text = raw_text.strip("```json").strip("```").strip()
            json_start = clean_text.find("{")
            json_end = clean_text.rfind("}")
            if json_start != -1 and json_end != -1:
                clean_text = clean_text[json_start:json_end + 1]
                reflection_json = json.loads(clean_text)
            else:
                reflection_json = None

            print("[INFO] Reflection generated successfully.")
            return reflection_json
        except Exception as e:
            print(f"Error generating reflection: {e}")
            return None

    # -------------------------------------------------------------------------
    # [추가] 트레이딩 의사결정 전에 "최신 reflection"을 system message에 반영하는 메서드
    def update_trading_model_prompt(self):
        """
        DB에서 최신 Reflection/Key Improvement/Confidence들을 불러와
        self.model_trading.system_instruction에 반영
        """
        # 1) 최근 Reflection 관련 정보 5개 불러오기
        reflections_texts = self.load_all_reflections(limit=5)
        # -> ["Reflection: ...\nKey Improvement Points: ...\nConfidence Level: ...", ...]

        # 2) 여러 개를 합쳐서 큰 문자열로 만들기
        joined_reflections = "\n\n".join(reflections_texts) if reflections_texts else "No recent reflections."

        # 3) 새 system message 구성
        new_system_msg = f"""
    You are an expert in Bitcoin trading, specializing in '워뇨띠'님의 매매법.
    You consider price trends, volume patterns, order book imbalances,
    market volatility, the Fear & Greed Index, relevant news, and the video transcript.

    Here is a transcript from the 워뇨띠's trading method video:
    {self.youtube_text}

    Here are previous trade reflections, key improvements, and confidence levels:
    {joined_reflections}

    Your job: Provide a buy/sell/hold decision & rationale.
        """.strip()

        # 4) 트레이딩 모델의 system_instruction 갱신
        self.model_trading.system_instruction = new_system_msg
        print("[INFO] Trading model system message updated with latest reflections.")

    # --------------------------------------------------------------------------------
    # Fear & Greed Index
    def get_fear_greed_index(self, limit=1):
        """Fear & Greed Index API (https://api.alternative.me/fng/)"""
        try:
            url = f"https://api.alternative.me/fng/?limit={limit}"
            response = requests.get(url, timeout=5)
            data = response.json()

            if "data" in data and len(data["data"]) > 0:
                latest = data["data"][0]
                return {
                    "value": latest.get("value"),
                    "classification": latest.get("value_classification"),
                    "timestamp": latest.get("timestamp"),
                }
            else:
                print("No fear & greed data available")
                return None
        except Exception as e:
            print(f"Error getting Fear and Greed Index: {e}")
            return None

    def get_investment_status(self):
        """잔고 조회 + 간단한 수익률 계산."""
        try:
            krw_balance = self.upbit.get_balance("KRW")
            btc_balance = self.upbit.get_balance(self.ticker)
            avg_buy_price = self.upbit.get_avg_buy_price(self.ticker)
            current_price = pyupbit.get_current_price(self.ticker)

            if (krw_balance is None or btc_balance is None
                or avg_buy_price is None or current_price is None):
                print("One or more balance/price values returned None. Skipping calculation.")
                return None

            total_value = krw_balance + (btc_balance * current_price)
            profit_loss = (current_price - avg_buy_price) * btc_balance if btc_balance > 0 else 0
            profit_loss_percentage = (profit_loss / total_value * 100) if total_value > 0 else 0

            print(f"[INFO] Investment status: KRW={krw_balance}, BTC={btc_balance}, "
                  f"avg_buy_price={avg_buy_price}, current_price={current_price}, "
                  f"total_value={total_value}, profit_loss={profit_loss}, "
                  f"profit_loss_percentage={profit_loss_percentage:.2f}%")
            
            return {
                "krw_balance": krw_balance,
                "btc_balance": btc_balance,
                "avg_buy_price": avg_buy_price,
                "current_price": current_price,
                "total_value": total_value,
                "profit_loss": profit_loss,
                "profit_loss_percentage": profit_loss_percentage
            }
        except Exception as e:
            print(f"Error getting investment status: {e}")
            return None

    def get_market_data(self):
        """주요 시장데이터 조회"""
        try:
            orderbook = pyupbit.get_orderbook(self.ticker)
            daily_ohlcv = pyupbit.get_ohlcv(self.ticker, interval="day", count=200)
            hourly_ohlcv = pyupbit.get_ohlcv(self.ticker, interval="minute60", count=48)
            print("[INFO] Market data fetched successfully.")
            return {
                "orderbook": orderbook,
                "daily_ohlcv": daily_ohlcv,
                "hourly_ohlcv": hourly_ohlcv
            }
        except Exception as e:
            print(f"Error getting market data: {e}")
            return None

    def add_custom_ta_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """ta 라이브러리를 활용해 기술지표 컬럼 추가."""
        df = dropna(df.copy())

        # EMA
        ema_indicator = EMAIndicator(close=df["close"], window=20)
        df["ema_20"] = ema_indicator.ema_indicator()

        # MACD
        macd_indicator = MACD(close=df["close"], window_slow=26, window_fast=12, window_sign=9)
        df["macd"] = macd_indicator.macd()
        df["macd_signal"] = macd_indicator.macd_signal()
        df["macd_diff"] = macd_indicator.macd_diff()

        # ADX
        adx_indicator = ADXIndicator(high=df["high"], low=df["low"], close=df["close"], window=14)
        df["adx"] = adx_indicator.adx()
        df["adx_pos"] = adx_indicator.adx_pos()
        df["adx_neg"] = adx_indicator.adx_neg()

        # RSI
        rsi_indicator = RSIIndicator(close=df["close"], window=14)
        df["rsi"] = rsi_indicator.rsi()

        # Stochastic
        stoch_indicator = StochasticOscillator(
            high=df["high"], 
            low=df["low"], 
            close=df["close"], 
            window=14, 
            smooth_window=3
        )
        df["stoch_k"] = stoch_indicator.stoch()
        df["stoch_d"] = stoch_indicator.stoch_signal()

        # Williams %R
        wr_indicator = WilliamsRIndicator(
            high=df["high"], 
            low=df["low"], 
            close=df["close"], 
            lbp=14
        )
        df["wr"] = wr_indicator.williams_r()

        # Bollinger Bands
        bb_indicator = BollingerBands(close=df["close"], window=20, window_dev=2)
        df["bb_mavg"] = bb_indicator.bollinger_mavg()
        df["bb_hband"] = bb_indicator.bollinger_hband()
        df["bb_lband"] = bb_indicator.bollinger_lband()

        # Keltner Channel
        kc_indicator = KeltnerChannel(high=df["high"], low=df["low"], close=df["close"], window=20)
        df["kc_hband"] = kc_indicator.keltner_channel_hband()
        df["kc_lband"] = kc_indicator.keltner_channel_lband()
        df["kc_mband"] = kc_indicator.keltner_channel_mband()

        # ATR
        atr_indicator = AverageTrueRange(high=df["high"], low=df["low"], close=df["close"], window=14)
        df["atr"] = atr_indicator.average_true_range()

        # OBV
        obv_indicator = OnBalanceVolumeIndicator(close=df["close"], volume=df["volume"])
        df["obv"] = obv_indicator.on_balance_volume()

        # VWAP
        vwap_indicator = VolumeWeightedAveragePrice(
            high=df["high"], low=df["low"], close=df["close"], volume=df["volume"], window=14
        )
        df["vwap"] = vwap_indicator.volume_weighted_average_price()

        return df

    def analyze_market_data(self, market_data):
        """일봉/시간봉 OHLCV, 오더북 등을 종합 분석."""
        if not market_data:
            return None

        analysis = {}

        # 일봉
        daily_df = market_data.get("daily_ohlcv")
        if isinstance(daily_df, pd.DataFrame) and not daily_df.empty:
            daily_with_indicators = self.add_custom_ta_features(daily_df.copy())
            daily_with_indicators = daily_with_indicators.iloc[-30:]  # 최근 30일
            analysis["daily_data"] = daily_with_indicators.to_dict()

        # 시간봉
        hourly_df = market_data.get("hourly_ohlcv")
        if isinstance(hourly_df, pd.DataFrame) and not hourly_df.empty:
            hourly_with_indicators = self.add_custom_ta_features(hourly_df.copy())
            hourly_with_indicators = hourly_with_indicators.iloc[-24:]  # 최근 24시간
            analysis["hourly_data"] = hourly_with_indicators.to_dict()

        # 오더북
        orderbook = market_data.get("orderbook")
        if (isinstance(orderbook, list) and len(orderbook) > 0
            and isinstance(orderbook[0], dict) and 'orderbook_units' in orderbook[0]):
            units = orderbook[0]['orderbook_units']
            if isinstance(units, list) and len(units) > 0:
                total_bid_size = sum([x.get('bid_size', 0) for x in units])
                total_ask_size = sum([x.get('ask_size', 0) for x in units])
                bid_ask_ratio = total_bid_size / total_ask_size if total_ask_size > 0 else 0

                analysis["order_book_analysis"] = {
                    "total_bid_size": total_bid_size,
                    "total_ask_size": total_ask_size,
                    "bid_ask_ratio": bid_ask_ratio
                }

        if not analysis:
            return None

        print("[INFO] Market data analysis completed.")
        return analysis

    def get_btc_news(self, query="BTC", gl="us", hl="en"):
        """SerpApi로 BTC 관련 뉴스 기사 정보를 가져옴."""
        if not self.serpapi_key:
            print("SERPAPI_KEY is not set. Please set it in your .env file.")
            return None

        try:
            serpapi_url = "https://serpapi.com/search.json"
            params = {
                "engine": "google_news",
                "q": query,
                "gl": gl,
                "hl": hl,
                "api_key": self.serpapi_key
            }
            response = requests.get(serpapi_url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            original_news_results = data.get("news_results", [])

            filtered_news_results = []
            for article in original_news_results:
                filtered_news_results.append({
                    "title": article.get("title"),
                    "date": article.get("date")
                })
            
            print(f"[INFO] Fetched {len(filtered_news_results)} news articles from SerpApi.")
            return filtered_news_results

        except Exception as e:
            print(f"Error fetching BTC news from SerpApi: {e}")
            return None

    def get_ai_decision(self, analysis, screenshot_path=None):
        """
        AI에게 트레이딩 의사결정을 요청.
        (buy/sell/hold) + 몇 %로 매매할지도(trade_percent) 알려주도록 프롬프트 예시를 제공.
        """
        if not analysis:
            return None

        converted_analysis = convert_timestamps(analysis)

        # 사용자 프롬프트
        user_prompt = f"""
Below is the current market analysis data. Please provide:
- A 'decision' (one of: buy / sell / hold)
- A 'reason' (string)
- A 'trade_percent' (integer, 0~100) for how much balance/holding to use

**Market Data**:
{json.dumps(converted_analysis, indent=2)}

Only return your answer in valid JSON format, for example:
{{
  "decision": "buy",
  "reason": "MACD is bullish, etc...",
  "trade_percent": 50
}}
""".strip()

        try:
            if screenshot_path and os.path.exists(screenshot_path):
                chart_image = Image.open(screenshot_path)
                response = self.model_trading.generate_content([user_prompt, chart_image])
            else:
                response = self.model_trading.generate_content([user_prompt])

            raw_text = response.candidates[0].content.parts[0].text
            clean_text = raw_text.strip("```json").strip("```").strip()

            json_start = clean_text.find("{")
            json_end = clean_text.rfind("}")
            if json_start != -1 and json_end != -1:
                clean_text = clean_text[json_start:json_end + 1]
                decision_json = json.loads(clean_text)
                print("[INFO] AI decision generated successfully.")
                return decision_json
            else:
                return None

        except Exception as e:
            print(f"Error getting AI decision: {e}")
            return None

    def execute_trade(self, decision, investment_status):
        """
        AI 결정에 따라 매수/매도/홀딩 수행
        decision: {
          "decision": "buy" / "sell" / "hold",
          "reason": "...",
          "trade_percent": int (0~100)
        }
        """
        if not decision or not investment_status:
            return

        try:
            trade_percent = decision.get("trade_percent", 100)
            if trade_percent < 0:
                trade_percent = 0
            if trade_percent > 100:
                trade_percent = 100

            if decision["decision"] == "buy":
                krw_balance = investment_status["krw_balance"]
                if krw_balance > 5000 and trade_percent > 0:
                    trade_amount = krw_balance * (trade_percent / 100.0) * 0.9995
                    if trade_amount > 5000:
                        print(f"[BUY] {trade_percent}% 매수 진행: {trade_amount} KRW")
                        self.log_trade(
                            action="buy",
                            trade_percent=trade_percent,
                            trade_amount=trade_amount,
                            reason=decision.get("reason", ""),
                            balance_after=krw_balance - trade_amount
                        )
                        # 실제 주문 예시:
                        self.upbit.buy_market_order(self.ticker, trade_amount)
                    else:
                        print("매수 가능한 금액이 부족하여 매수 생략")

            elif decision["decision"] == "sell":
                btc_balance = investment_status["btc_balance"]
                current_price = investment_status["current_price"]
                if btc_balance * current_price > 5000 and trade_percent > 0:
                    sell_amount_btc = btc_balance * (trade_percent / 100.0)
                    trade_amount = sell_amount_btc * current_price
                    if trade_amount > 5000:
                        print(f"[SELL] {trade_percent}% 매도 진행: {sell_amount_btc} BTC")
                        self.log_trade(
                            action="sell",
                            trade_percent=trade_percent,
                            trade_amount=trade_amount,
                            reason=decision.get("reason", ""),
                            balance_after=investment_status["krw_balance"] + trade_amount
                        )
                        # 실제 주문 예시:
                        self.upbit.sell_market_order(self.ticker, sell_amount_btc)
                    else:
                        print("매도 가능한 수량이 부족하여 매도 생략")

            elif decision["decision"] == "hold":
                print(f"[HOLD] {decision.get('reason', '')}")
                self.log_trade(
                    action="hold",
                    trade_percent=0,
                    trade_amount=0,
                    reason=decision.get("reason", ""),
                    balance_after=investment_status["krw_balance"]
                )

        except Exception as e:
            print(f"Error executing trade: {e}")

    def run(self):
        """메인 트레이딩 루프."""
        while True:
            try:
                # 1) 현재 잔고/투자상태
                investment_status = self.get_investment_status()
                if investment_status is None:
                    time.sleep(60)
                    continue

                # 2) 시장 데이터 수집 & 분석
                market_data = self.get_market_data()
                analysis = self.analyze_market_data(market_data)
                if analysis is None:
                    time.sleep(60)
                    continue

                # 3) Fear & Greed Index
                fng_data = self.get_fear_greed_index(limit=1)
                if fng_data:
                    analysis["fear_and_greed_index"] = fng_data

                # 4) BTC 뉴스
                btc_news = self.get_btc_news(query="BTC", gl="us", hl="en")
                if btc_news:
                    analysis["btc_news"] = btc_news

                # 5) 차트 스크린샷
                screenshot_name = "upbit_chart.png"
                chart_url = "https://upbit.com/full_chart?code=CRIX.UPBIT.KRW-BTC"
                take_fullpage_screenshot(chart_url, screenshot_name)

                # 6) 트레이딩 의사결정 전, 시스템 메시지(Reflection) 갱신
                self.update_trading_model_prompt()

                # 7) 트레이딩 의사결정
                decision = self.get_ai_decision(analysis, screenshot_path=screenshot_name)
                if decision:
                    self.execute_trade(decision, investment_status)
                else:
                    print("No valid AI decision.")

                # 8) 매매 후 → 반성문 생성
                self.db_cursor.execute("SELECT * FROM trade_records WHERE coin_ticker = ? ORDER BY timestamp DESC LIMIT 1", (self.ticker,))
                recent_trades = self.db_cursor.fetchall()
                reflection_json = self.reflect_on_trades_and_market(recent_trades, analysis)
                if reflection_json and "reflection" in reflection_json:
                    latest_trade_id = recent_trades[0][0]  # 가장 최근 매매 id
                    self.store_reflection_for_trade(latest_trade_id, reflection_json)

                print("\nWaiting for next trading cycle...")
                time.sleep(60 * 60 * 6)  # 6시간마다 실행

            except Exception as e:
                print(f"Error in main loop: {e}")
                time.sleep(60)


if __name__ == "__main__":
    trader = BitcoinTrader()
    trader.run()
