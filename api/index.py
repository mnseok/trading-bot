from flask import Flask, render_template
import sqlite3
import pandas as pd
import socket
import subprocess

app = Flask(__name__)

# SQLite database path
DB_PATH = "trading_records.db"

def fetch_trade_records():
    """Fetch trading records from SQLite."""
    try:
        conn = sqlite3.connect(DB_PATH)
        query = "SELECT * FROM trade_records ORDER BY timestamp DESC"
        df = pd.read_sql_query(query, conn)
        conn.close()
        return df
    except Exception as e:
        print(f"Error fetching trade records: {e}")
        return pd.DataFrame()

def get_local_ip():
    """Get the local IP address of the machine."""
    try:
        hostname = socket.gethostname()
        local_ip = socket.gethostbyname(hostname)
        return local_ip
    except Exception as e:
        print(f"Error getting local IP: {e}")
        return "Unavailable"

@app.route("/")
def home():
    """Homepage displaying the trading records and IP address."""
    records = fetch_trade_records()
    local_ip = get_local_ip()

    if records.empty:
        message = "No trade records found."
        return render_template("index.html", message=message, local_ip=local_ip)

    # Convert DataFrame to HTML table
    table_html = records.to_html(index=False, classes="table table-striped", border=0)
    return render_template("index.html", table_html=table_html, local_ip=local_ip)

def start_trading_bot():
    """Ensure the trading bot script is running."""
    try:
        # Replace 'trading_bot.py' with the actual filename of your bot script
        subprocess.Popen(["python", "trading_bot.py"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print("Trading bot script started.")
    except Exception as e:
        print(f"Error starting trading bot: {e}")

if __name__ == "__main__":
    # Start the trading bot in the background
    start_trading_bot()
    
    # Run the Flask app
    app.run(debug=True)
