import sys
import json
import os
from typing import Optional, Dict, Any, List, Tuple
import time
import ccxt
import numpy as np
import pandas as pd

CONFIG_FILE = "config.json"

DEFAULT_CONFIG = {
    "api_key": "",
    "secret_key": "",
    "password": "",
    "proxies": {"http": "socks5h://127.0.0.1:7890", "https": "socks5h://127.0.0.1:7890"},
    "base_currency": "BTC",
    "leverage": 50,
    "contract_size": 10,
    "take_profit_percent": 25,
    "stop_loss_percent": 50,
    "timeframe": "15m",
    "enable_long": True,
    "enable_short": True,
    "cooldown_minutes": 15,
    "position_check_interval": 0.1
}



class StrategyWorker:
    def __init__(self, config):
        self.api_key = config["api_key"]
        self.secret_key = config["secret_key"]
        self.password = config["password"]
        self.proxies = config.get("proxies", {})
        self.base_currency = config["base_currency"]
        self.symbol = f"{self.base_currency}-USDT-SWAP"
        self.leverage = config["leverage"]
        self.contract_size = float(config["contract_size"])
        self.take_profit_percent = config["take_profit_percent"]
        self.stop_loss_percent = config["stop_loss_percent"]
        self.timeframe = config["timeframe"]
        self.enable_long = config["enable_long"]
        self.enable_short = config["enable_short"]
        self.running = False
        self.cooldown_seconds = self._convert_timeframe_to_seconds(self.timeframe)
        self.position_check_interval = config["position_check_interval"]
        self.last_close_time = 0

    def _convert_timeframe_to_seconds(self, tf):
        minute_map = {
            '1m': 60,
            '5m': 300,
            '15m': 900,
            '30m': 1800,
            '1h': 3600,
            '4h': 14400
        }
        return minute_map.get(tf, 900)

    def run(self):
        self.running = True
        exchange = ccxt.okx({
            'enableRateLimit': True,
            'apiKey': self.api_key,
            'secret': self.secret_key,
            'password': self.password,
            'proxies': self.proxies,
            'options': {
                'defaultType': 'swap',
                'fetchBalance': False,
                'adjustForTimeDifference': True,
            }
        })

        try:
            exchange.load_markets()
            print("✅ 已连接至 OKX 永续合约市场")
        except Exception as e:
            print(f"❌ 连接失败: {str(e)}")
            return

        in_position, direction, entry_price, amount = self.get_current_position(exchange)
        if in_position:
            print(f"🔄 发现已有持仓：{direction}单 @ {entry_price}")

        while self.running:
            try:
                df = self.fetch_kline(exchange, self.symbol, self.timeframe, 100)
                if df.empty:
                    print("⚠️ K线数据为空，等待下一轮...")
                    time.sleep(5)
                    continue

                latest_row = df.iloc[0]
                current_open = latest_row['open']
                current_high = latest_row['high']
                current_low = latest_row['low']
                print(f"\n📈 当前行情 | 开盘价: {current_open}, 最高价: {current_high}, 最低价: {current_low}")

                df_for_indicator = df.copy().sort_index(ascending=True)
                kc_upper, _, kc_lower = self.calculate_keltner_channel(df_for_indicator)
                if kc_upper is None or kc_lower is None:
                    print("⚠️ Keltner Channel 指标计算失败")
                    time.sleep(1)
                    continue

                upper = kc_upper.iloc[-1]
                lower = kc_lower.iloc[-1]
                print(f"📊 布林带通道: 上轨={upper:.2f}, 下轨={lower:.2f}")

                trend = self.detect_trend(df)
                print(f"🔍 当前趋势判断: {trend}")

                current_time = time.time()

                if not in_position and (current_time - self.last_close_time) > self.cooldown_seconds:
                    previous_row = df.iloc[1]
                    prev_open = previous_row['open']
                    prev_close = previous_row['close']
                    prev_is_green = prev_close > prev_open

                    if self.enable_long and prev_is_green and current_open > upper:
                        order = self.place_long_order(exchange, self.contract_size, current_open)
                        if order:
                            fill_price = order.get('price') or order.get('average', order.get('fillPrice'))
                            if not fill_price:
                                ticker = exchange.fetch_ticker(self.symbol)
                                fill_price = ticker['last']
                            in_position = True
                            direction = 'long'
                            entry_price = float(fill_price)
                            print(f"🟢 多单已建立 @ {entry_price:.2f}")

                    elif self.enable_short and not prev_is_green and current_open < lower:
                        order = self.place_short_order(exchange, self.contract_size, current_open)
                        if order:
                            fill_price = order.get('price') or order.get('average', order.get('fillPrice'))
                            if not fill_price:
                                ticker = exchange.fetch_ticker(self.symbol)
                                fill_price = ticker['last']
                            in_position = True
                            direction = 'short'
                            entry_price = float(fill_price)
                            print(f"🔴 空单已建立 @ {entry_price:.2f}")

                if in_position:
                    ticker = exchange.fetch_ticker(self.symbol)
                    current_price = ticker['last']
                    if direction == 'long':
                        current_return = ((current_price - entry_price) / entry_price) * self.leverage * 100
                        profit_percent = f"{current_return:.2f}%"
                        print(f"💼 当前多单收益: {profit_percent}")
                        if current_return >= self.take_profit_percent or current_return <= -self.stop_loss_percent:
                            order = self.place_close_long_order(exchange, amount, current_price)
                            if order:
                                profit = ((current_price - entry_price) / entry_price) * amount * self.leverage
                                print(f"✅ 平多单成功，收益: ${profit:.2f}")
                                in_position = False
                                direction = None
                                self.last_close_time = time.time()
                    elif direction == 'short':
                        current_return = ((entry_price - current_price) / entry_price) * self.leverage * 100
                        profit_percent = f"{current_return:.2f}%"
                        print(f"💼 当前空单收益: {profit_percent}")
                        if current_return >= self.take_profit_percent or current_return <= -self.stop_loss_percent:
                            order = self.place_close_short_order(exchange, amount, current_price)
                            if order:
                                profit = ((entry_price - current_price) / entry_price) * amount * self.leverage
                                print(f"✅ 平空单成功，收益: ${profit:.2f}")
                                in_position = False
                                direction = None
                                self.last_close_time = time.time()

                time.sleep(self.position_check_interval)
            except Exception as e:
                print(f"❌ 错误: {str(e)}")
                time.sleep(5)

    def stop(self):
        self.running = False

    def fetch_kline(self, exchange, symbol='BTC-USDT-SWAP', timeframe='15m', limit=100) -> pd.DataFrame:
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            df = df.sort_index(ascending=False).head(200)
            return df
        except Exception as e:
            self.log_signal.emit(f"⚠️ 获取K线失败: {str(e)}")
            return pd.DataFrame()

    def calculate_atr(self, df, period=14) -> pd.Series:
        high = df['high']
        low = df['low']
        close = df['close']
        tr0 = abs(high - low)
        tr1 = abs(high - close.shift())
        tr2 = abs(low - close.shift())
        tr = pd.concat([tr0, tr1, tr2], axis=1).max(axis=1)
        atr = tr.ewm(span=period, min_periods=period + 1).mean()
        return atr

    def calculate_keltner_channel(self, df, period=20, mult=1.5) -> Tuple[pd.Series, ...]:
        if len(df) < period:
            return None, None, None
        middle = df['close'].ewm(span=period).mean()
        atr = self.calculate_atr(df, period=14)
        upper = middle + mult * atr
        lower = middle - mult * atr
        return upper, middle, lower

    def calculate_adx(self, df, period=14) -> pd.Series:
        high = df['high']
        low = df['low']
        close = df['close']
        plus_dm = high.diff()
        minus_dm = low.diff()
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm > 0] = 0
        tr = pd.concat([high - low, abs(high - close.shift()), abs(low - close.shift())], axis=1).max(axis=1)
        atr = tr.ewm(span=period, min_periods=period + 1).mean()
        plus_di = 100 * (plus_dm.rolling(period).sum() / atr)
        minus_di = abs(100 * (minus_dm.rolling(period).sum() / atr))
        dx = 100 * abs((plus_di - minus_di) / (plus_di + minus_di))
        adx = dx.rolling(period).mean()
        return adx

    def calculate_ema(self, series, period) -> pd.Series:
        return series.ewm(span=period, adjust=False).mean()

    def calculate_supertrend(self, df, period=10, multiplier=3) -> Tuple[pd.Series, bool]:
        try:
            hl2 = (df['high'] + df['low']) / 2
            atr = self.calculate_atr(df, period=period)
            upperband = hl2 + multiplier * atr
            lowerband = hl2 - multiplier * atr

            in_uptrend = True
            super_trend = pd.Series(index=df.index)

            for i in range(len(df)):
                if i == 0:

                    super_trend.iloc[i] = lowerband.iloc[i]
                    continue

                close = df['close'].iloc[i]
                prev_upper = upperband.iloc[i - 1]
                prev_lower = lowerband.iloc[i - 1]

                if close > prev_upper:
                    super_trend.iloc[i] = lowerband.iloc[i]
                    in_uptrend = True
                elif close < prev_lower:
                    super_trend.iloc[i] = upperband.iloc[i]
                    in_uptrend = False
                else:
                    super_trend.iloc[i] = lowerband.iloc[i] if in_uptrend else upperband.iloc[i]

            return super_trend, in_uptrend
        except Exception as e:
            self.log_signal.emit(f"❌ SuperTrend计算异常: {str(e)}")
            return pd.Series(), False

    def detect_trend(self, df) -> str:
        if len(df) < 50:
            self.log_signal.emit("⚠️ 数据不足，无法判断趋势")
            return "sideways"
        close = df['close']
        adx = self.calculate_adx(df)
        trend_strength = adx.iloc[-1] > 25
        macd_line, signal_line, _ = self.calculate_macd(df)
        macd_bullish = macd_line.iloc[-1] > signal_line.iloc[-1]
        ema_short = self.calculate_ema(close, 10)
        ema_long = self.calculate_ema(close, 50)
        price_above_ema_short = close.iloc[-1] > ema_short.iloc[-1]
        ema_short_above_long = ema_short.iloc[-1] > ema_long.iloc[-1]
        try:
            _, supertrend_uptrend = self.calculate_supertrend(df)
        except Exception as e:
            self.log_signal.emit(f"⚠️ SuperTrend 异常: {str(e)}")
            supertrend_uptrend = None
        bullish_signals = sum([
            trend_strength,
            macd_bullish,
            price_above_ema_short,
            ema_short_above_long,
            supertrend_uptrend
        ])
        bearish_signals = sum([
            trend_strength,
            not macd_bullish,
            not price_above_ema_short,
            not ema_short_above_long,
            not supertrend_uptrend if supertrend_uptrend is not None else False
        ])
        if bullish_signals >= 3:
            return "up"
        elif bearish_signals >= 3:
            return "down"
        else:
            return "sideways"

    def calculate_macd(self, df, fast_period=12, slow_period=26, signal_period=9) -> Tuple[
        pd.Series, pd.Series, pd.Series]:
        close = df['close']
        ema_fast = close.ewm(span=fast_period, min_periods=fast_period).mean()
        ema_slow = close.ewm(span=slow_period, min_periods=slow_period).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal_period, min_periods=signal_period).mean()
        macd_hist = macd_line - signal_line
        return macd_line, signal_line, macd_hist

    def set_leverage(self, exchange, leverage=50) -> None:
        try:
            params = {
                'instId': self.symbol,
                'lever': str(leverage),
                'mgnMode': 'cross'
            }
            exchange.private_post_account_set_position_mode({'posMode': 'net_mode'})
            exchange.private_post_account_set_leverage(params)
            self.log_signal.emit("✅ 杠杆设置成功")
        except Exception as e:
            self.log_signal.emit(f"⚠️ 设置杠杆失败: {str(e)}")

    def format_amount(self, exchange, amount) -> str:
        market = exchange.market(self.symbol)
        precision = market['precision']['amount']
        min_amount = market['limits']['amount']['min']
        decimal_places = int(round(-np.log10(precision)))
        amount = max(amount, min_amount)
        return "{:.{}f}".format(amount, decimal_places)

    def format_price(self, exchange, price) -> str:
        market = exchange.market(self.symbol)
        precision = market['precision']['price']
        decimal_places = int(round(-np.log10(precision)))
        return "{:.{}f}".format(price, decimal_places)

    def place_long_order(self, exchange, amount, price) -> Optional[dict]:
        try:
            exchange.load_markets()
            amount_str = self.format_amount(exchange, amount)


            order = exchange.create_order(
                symbol=self.symbol,
                type='market',
                side='buy',
                amount=float(amount_str),

                params={'tdMode': 'cross'}
            )
            fill_price = order.get('price') or order.get('average', order.get('fillPrice'))
            if not fill_price:
                ticker = exchange.fetch_ticker(self.symbol)
                fill_price = ticker['last']
            self.log_signal.emit(f"📈 成功做多 @ {fill_price:.2f}, 数量: {amount_str}")
            return order
        except Exception as e:
            error_msg = str(e).lower()
            if "insufficient" in error_msg:
                self.log_signal.emit("🚫 做多失败：账户余额不足")
            elif "price is lower than" in error_msg:
                self.log_signal.emit("⚠️ 做多失败：下单价格低于最低限制，尝试提高价格")
            else:
                self.log_signal.emit(f"❌ 【做多失败】错误: {str(e)}")
            return None

    def place_short_order(self, exchange, amount, price) -> Optional[dict]:
        try:
            exchange.load_markets()
            amount_str = self.format_amount(exchange, amount)
            order = exchange.create_order(
                symbol=self.symbol,
                type='market',
                side='sell',
                amount=float(amount_str),
                params={'tdMode': 'cross'}
            )
            fill_price = order.get('price') or order.get('average', order.get('fillPrice'))
            if not fill_price:
                ticker = exchange.fetch_ticker(self.symbol)
                fill_price = ticker['last']
            self.log_signal.emit(f"📉 成功做空 @ {fill_price:.2f}, 数量: {amount_str}")
            return order
        except Exception as e:
            error_msg = str(e).lower()
            if "insufficient" in error_msg:
                self.log_signal.emit("🚫 做空失败：账户余额不足")
            elif "price is higher than" in error_msg:
                self.log_signal.emit("⚠️ 做空失败：下单价格高于当前市场价上限，尝试降低价格")
            else:
                self.log_signal.emit(f"❌ 【做空失败】错误: {str(e)}")
            return None

    def place_close_long_order(self, exchange, amount, price) -> Optional[dict]:
        try:
            exchange.load_markets()
            amount_str = self.format_amount(exchange, amount)
            order = exchange.create_order(
                symbol=self.symbol,
                type='market',
                side='sell',
                amount=float(amount_str),
                params={'tdMode': 'cross'}
            )
            fill_price = order.get('price') or order.get('average', order.get('fillPrice'))
            if not fill_price:
                ticker = exchange.fetch_ticker(self.symbol)
                fill_price = ticker['last']
            self.log_signal.emit(f"✅ 平多单 @ {fill_price:.2f}, 数量: {amount_str}")
            return order
        except Exception as e:
            self.log_signal.emit(f"❌ 【平多失败】错误: {str(e)}")
            return None

    def place_close_short_order(self, exchange, amount, price) -> Optional[dict]:
        try:
            exchange.load_markets()
            amount_str = self.format_amount(exchange, amount)
            order = exchange.create_order(
                symbol=self.symbol,
                type='market',
                side='buy',
                amount=float(amount_str),
                params={'tdMode': 'cross'}
            )
            fill_price = order.get('price') or order.get('average', order.get('fillPrice'))
            if not fill_price:
                ticker = exchange.fetch_ticker(self.symbol)
                fill_price = ticker['last']
            self.log_signal.emit(f"✅ 平空单 @ {fill_price:.2f}, 数量: {amount_str}")
            return order
        except Exception as e:
            self.log_signal.emit(f"❌ 【平空失败】错误: {str(e)}")
            return None

    def get_current_position(self, exchange) -> Tuple[bool, str, float, float]:
        try:
            positions = exchange.fetch_positions([self.symbol])
            # print("Positions:", positions)

            for pos in positions:

                symbol_positions = pos['symbol'].replace('/', '-').replace(':USDT', '-SWAP')
                if symbol_positions == self.symbol and float(pos['contracts']) != 0:
                    side = pos['side'].lower()
                    entry_price = float(pos['entryPrice'])
                    amount = float(pos['contracts'])

                    print(f"发现持仓: {side}单 @ {entry_price}, 数量: {amount}")
                    return True, side, entry_price, amount

            return False, '', 0.0, 0.0

        except Exception as e:
            self.log_signal.emit(f"⚠️ 获取持仓失败: {str(e)}")
            return False, '', 0.0, 0.0


def load_config():
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, 'r') as f:
            return json.load(f)
    return DEFAULT_CONFIG

def save_config(config):
    with open(CONFIG_FILE, 'w') as f:
        json.dump(config, f, indent=4)

def main():
    print("🚀 加密货币交易系统 - 命令行版")

    config = load_config()
    print("📁 当前配置:")
    for k, v in config.items():
        print(f"  {k}: {v}")

    choice = input("\n是否使用当前配置启动策略？(y/n): ").strip().lower()
    if choice != 'y':
        print("🚫 退出程序")
        return

    worker = StrategyWorker(config)

    try:
        print("▶️ 策略开始运行... 按 Ctrl+C 停止")
        worker.run()
    except KeyboardInterrupt:
        print("\n⏹ 用户中断操作")
        worker.stop()

if __name__ == '__main__':
    main()