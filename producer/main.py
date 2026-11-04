import json
import websocket
import os
from kafka import KafkaProducer
from datetime import datetime, timezone

# set up redpanda connection to localhost: 9092
# redpanda only accepts data in raw byte form so the dictionary v is converted into raw utf-8
producer = KafkaProducer(
    bootstrap_servers=os.getenv('BOOTSTRAP_SERVERS', 'localhost:9092').split(','),
    value_serializer=lambda v: json.dumps(v).encode('utf-8')
)
topic_name = 'real-market-trades'

# when coinbase sends back a trade we check if the type is match to make sure it's a trade and not noise
# we then create a payload (dictionary) with all the trade information and send it to redpanda and print it out to console
def on_message(ws, message):

    try:
        data = json.loads(message)

        if data.get('type') == 'match':
            # normalize the data 

            raw_time = data.get('time')
            parsed = datetime.fromisoformat(raw_time.replace('Z', '+00:00'))
            formatted = parsed.strftime('%Y-%m-%d %H:%M:%S.') + f"{parsed.microsecond // 1000:03d}"

            trade_event = {
                'trade_id': data.get('trade_id'),
                'symbol': data.get('product_id'),        # e.g., "BTC-USD"
                'price': float(data.get('price')),       # execution price
                'volume': float(data.get('size')),       # amount traded
                'side': data.get('side'),                # "buy" or "sell"
                'trade_time': formatted                  # real trade timestamp
            }

            # Dispatch the real-world trade to Redpanda
            producer.send(topic_name, value=trade_event)
            print(f"Real Trade Forwarded: {trade_event}")

    except Exception as e:
        print(f"Error processing message: {e}")

# establish a connection with coinbase and send a payload resquesting completed (matches) trade information on specified coins (product_ids)
def on_open(ws):
    print("Connected to Coinbase WebSocket!")

    subscribe_message = {
        "type": "subscribe",
        "product_ids": ["BTC-USD", "ETH-USD", "SOL-USD"],
        "channels": ["matches"]
    }
    ws.send(json.dumps(subscribe_message))
    print(f"Subscribed to trade channels. Streaming to Redpanda '{topic_name}'...")


def on_error(ws, error):
    print(f"WebSocket Error: {error}")


def on_close(ws, close_status_code, close_msg):
    print("WebSocket connection closed.")
    producer.flush()
    producer.close()

# get the coinbase websocket url, then create a websocket with the socket giving it 
# the previously created functions for different events
if __name__ == "__main__":
    # coinbase websocket endpoint
    socket_url = "wss://ws-feed.exchange.coinbase.com"

    # create a websocket 
    ws = websocket.WebSocketApp(
        socket_url,
        on_open=on_open,
        on_message=on_message,
        on_error=on_error,
        on_close=on_close
    )

    # call run_forever method from our websocket to continually listen to trades and send them to red panda
    try:
        ws.run_forever()
    except KeyboardInterrupt:
        print("\nShutting down stream...")

