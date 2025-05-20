import sys
import json
import os
from typing import Optional, Dict, Any, List, Tuple
import time
import ccxt
import numpy as np
import pandas as pd
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QLineEdit, QPushButton,
    QTextEdit, QVBoxLayout, QHBoxLayout, QGroupBox, QFormLayout,
    QComboBox, QCheckBox, QTextBrowser
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QFont

CONFIG_FILE = "config.json"


def load_config() -> dict:
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, 'r') as f:
            return json.load(f)
    return {
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
        "position_check_interval": 0.1  # 默认时间间隔
    }


def save_config(config: dict) -> None:
    with open(CONFIG_FILE, 'w') as f:
        json.dump(config, f, indent=4)


class StrategyWorker(QThread):
    log_signal = pyqtSignal(str)
    update_price_signal = pyqtSignal(float, float, float)
    update_bands_signal = pyqtSignal(float, float)
    update_position_signal = pyqtSignal(str, str, float, str)

    def __init__(
            self,
            api_key: str,
            secret_key: str,
            password: str,
            proxies: Optional[Dict[str, str]] = None,
            base_currency: str = 'BTC',
            leverage: int = 50,
            contract_size: float = 10,
            take_profit_percent: int = 25,
            stop_loss_percent: int = 50,
            timeframe: str = '15m',
            cooldown_minutes: int = 15,
            enable_long: bool = True,
            enable_short: bool = True,
            position_check_interval: float = 0.1
    ):
        super().__init__()
        self.api_key = api_key
        self.secret_key = secret_key
        self.password = password
        self.proxies = proxies or {'http': 'socks5h://127.0.0.1:7890', 'https': 'socks5h://127.0.0.1:7890'}
        self.base_currency = base_currency
        self.symbol = f"{base_currency}-USDT-SWAP"
        self.leverage = leverage
        self.contract_size = float(contract_size)
        self.take_profit_percent = take_profit_percent
        self.stop_loss_percent = stop_loss_percent
        self.timeframe = timeframe
        self.enable_long = enable_long
        self.enable_short = enable_short
        self.running = False
        self.cooldown_minutes = cooldown_minutes
        self.position_check_interval = position_check_interval
        self.cooldown_seconds = self._convert_timeframe_to_seconds(timeframe)
        self.last_close_time = 0

    def _convert_timeframe_to_seconds(self, tf: str) -> int:
        minute_map = {
            '1m': 60,
            '5m': 300,
            '15m': 900,
            '30m': 1800,
            '1h': 3600,
            '4h': 14400
        }
        return minute_map.get(tf, 900)

    def run(self) -> None:
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
            self.log_signal.emit("✅ 已连接至 OKX 永续合约市场")
        except Exception as e:
            self.log_signal.emit(f"❌ 连接失败: {str(e)}")
            return

        # 检查当前持仓
        in_position, direction, entry_price, amount = self.get_current_position(exchange)


        if in_position:
            self.log_signal.emit(f"🔄 发现已有持仓：{direction}单 @ {entry_price}")
            self.update_position_signal.emit("已持仓", direction.upper(), entry_price, "--%")
            while self.running:
                ticker = exchange.fetch_ticker(self.symbol)
                current_price = ticker['last']
                if direction == 'long':
                    current_return = ((current_price - entry_price) / entry_price) * self.leverage * 100
                    profit_percent = f"{current_return:.2f}%"
                    self.update_position_signal.emit("已持仓", "多单", entry_price, profit_percent)
                    if current_return >= self.take_profit_percent or current_return <= -self.stop_loss_percent:
                        order = self.place_close_long_order(exchange, amount, current_price)
                        if order:
                            profit = ((current_price - entry_price) / entry_price) * amount * self.leverage
                            self.log_signal.emit(f"✅ 平多单成功，收益: ${profit:.2f}")
                            in_position = False
                            direction = None
                            self.last_close_time = time.time()
                            self.update_position_signal.emit("未持仓", "--", 0, "--%")
                            self.running = False

                elif direction == 'short':
                    current_return = ((entry_price - current_price) / entry_price) * self.leverage * 100
                    profit_percent = f"{current_return:.2f}%"
                    self.update_position_signal.emit("已持仓", "空单", entry_price, profit_percent)
                    if current_return >= self.take_profit_percent or current_return <= -self.stop_loss_percent:
                        order = self.place_close_short_order(exchange, amount, current_price)
                        if order:
                            profit = ((entry_price - current_price) / entry_price) * amount * self.leverage
                            self.log_signal.emit(f"✅ 平空单成功，收益: ${profit:.2f}")
                            in_position = False
                            direction = None
                            self.last_close_time = time.time()
                            self.update_position_signal.emit("未持仓", "--", 0, "--%")
                            self.running = False

                time.sleep(self.position_check_interval)

        self.running = True
        self.set_leverage(exchange, self.leverage)
        while self.running:
            try:
                df = self.fetch_kline(exchange, self.symbol, self.timeframe, 100)
                if df.empty:
                    self.log_signal.emit("⚠️ K线数据为空，等待下一轮...")
                    time.sleep(5)
                    continue

                latest_row = df.iloc[0]
                current_open = latest_row['open']
                current_high = latest_row['high']
                current_low = latest_row['low']
                self.update_price_signal.emit(current_open, current_high, current_low)

                df_for_indicator = df.copy().sort_index(ascending=True)
                kc_upper, _, kc_lower = self.calculate_keltner_channel(df_for_indicator)
                if kc_upper is None or kc_lower is None:
                    self.log_signal.emit("⚠️ Keltner Channel 指标计算失败")
                    time.sleep(1)
                    continue

                upper = kc_upper.iloc[-1]
                lower = kc_lower.iloc[-1]
                self.update_bands_signal.emit(upper, lower)

                current_time = time.time()
                trend = self.detect_trend(df)
                if trend == "sideways":
                    continue

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
                            self.update_position_signal.emit("已持仓", "多单", entry_price, "--%")
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
                            self.update_position_signal.emit("已持仓", "空单", entry_price, "--%")

                if in_position:
                    ticker = exchange.fetch_ticker(self.symbol)
                    current_price = ticker['last']
                    if direction == 'long':
                        current_return = ((current_price - entry_price) / entry_price) * self.leverage * 100
                        profit_percent = f"{current_return:.2f}%"
                        self.update_position_signal.emit("已持仓", "多单", entry_price, profit_percent)
                        if current_return >= self.take_profit_percent or current_return <= -self.stop_loss_percent:
                            order = self.place_close_long_order(exchange, amount, current_price)
                            if order:
                                profit = ((current_price - entry_price) / entry_price) * amount * self.leverage
                                self.log_signal.emit(f"✅ 平多单成功，收益: ${profit:.2f}")
                                in_position = False
                                direction = None
                                self.last_close_time = time.time()
                                self.update_position_signal.emit("未持仓", "--", 0, "--%")
                    elif direction == 'short':
                        current_return = ((entry_price - current_price) / entry_price) * self.leverage * 100
                        profit_percent = f"{current_return:.2f}%"
                        self.update_position_signal.emit("已持仓", "空单", entry_price, profit_percent)
                        if current_return >= self.take_profit_percent or current_return <= -self.stop_loss_percent:
                            order = self.place_close_short_order(exchange, amount, current_price)
                            if order:
                                profit = ((entry_price - current_price) / entry_price) * amount * self.leverage
                                self.log_signal.emit(f"✅ 平空单成功，收益: ${profit:.2f}")
                                in_position = False
                                direction = None
                                self.last_close_time = time.time()
                                self.update_position_signal.emit("未持仓", "--", 0, "--%")

                time.sleep(self.position_check_interval)
            except Exception as e:
                self.log_signal.emit(f"❌ 错误: {str(e)}")
                time.sleep(5)

    def stop(self) -> None:
        self.running = False
        self.quit()
        self.wait()

    def fetch_kline(self, exchange, symbol='BTC-USDT-SWAP', timeframe='15m', limit=100) -> pd.DataFrame:
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            df = df.sort_index(ascending=False).head(200)  # 限制最大行数
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
                    # 初始状态使用第一个值
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
            # 移除 price_str，不再需要
            order = exchange.create_order(
                symbol=self.symbol,
                type='market',  # 改为市价单
                side='buy',
                amount=float(amount_str),
                # 不传 price
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
            # print("Positions:", positions)  # 调试输出持仓信息

            for pos in positions:
                # 确保持仓的合约符号与目标合约一致
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


class TradingStrategyGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Crazy Marxism 加密货币交易系统")
        self.resize(1600, 600)

        self.setFont(QFont("Segoe UI", 10))
        # Apply dark theme
        self.setStyleSheet("""
            QWidget {
                background-color: #1e1e2f;
                color: #ffffff;
                font-family: 'Segoe UI';
            }
            QLabel {
                color: #cccccc;
            }
            QLineEdit, QComboBox, QTextEdit {
                padding: 5px;
                border: 1px solid #444;
                border-radius: 4px;
                background-color: #2d2d3d;
                color: #ffffff;
            }
            QPushButton {
                background-color: #3a3aff;
                color: white;
                border: none;
                padding: 6px 12px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #5555ff;
            }
            QGroupBox {
                border: 1px solid #444;
                border-radius: 4px;
                margin-top: 10px;
                padding: 10px;
            }
            QGroupBox::title {
                subline-control-position: top center;
                padding: 0 3px;
            }
            QTextEdit {
                background-color: #1a1a2e;
                border: 1px solid #333;
            }
        """)
        main_layout = QHBoxLayout(self)
        main_layout.setSpacing(20)
        main_layout.setContentsMargins(20, 20, 20, 20)

        left_panel = QGroupBox("参数设置")
        left_layout = QFormLayout()

        self.api_key_input = QLineEdit("")
        self.secret_key_input = QLineEdit("")
        self.password_input = QLineEdit("")
        self.proxy_http_input = QLineEdit("socks5h://127.0.0.1:7890")
        self.proxy_https_input = QLineEdit("socks5h://127.0.0.1:7890")
        self.base_currency_input = QLineEdit("BTC")
        self.timeframe_combo = QComboBox()
        self.timeframe_combo.addItems(["1m", "5m", "15m", "30m", "1h", "4h"])
        self.timeframe_combo.setCurrentText("15m")
        self.leverage_input = QLineEdit("50")
        self.contract_size_input = QLineEdit("10")  # 使用张数
        self.take_profit_input = QLineEdit("25")
        self.stop_loss_input = QLineEdit("50")
        self.cooldown_input = QLineEdit("15")
        self.position_check_input = QLineEdit("0.1")  # 用户自定义时间间隔
        self.long_checkbox = QCheckBox("✅ 允许做多")
        self.short_checkbox = QCheckBox("✅ 允许做空")
        self.long_checkbox.setChecked(True)
        self.short_checkbox.setChecked(True)

        config = load_config()
        self.api_key_input.setText(config.get('api_key', ''))
        self.secret_key_input.setText(config.get('secret_key', ''))
        self.password_input.setText(config.get('password', ''))
        self.proxy_http_input.setText(config.get('proxies', {}).get('http', 'socks5h://127.0.0.1:7890'))
        self.proxy_https_input.setText(config.get('proxies', {}).get('https', 'socks5h://127.0.0.1:7890'))
        self.base_currency_input.setText(config.get('base_currency', 'BTC'))
        self.timeframe_combo.setCurrentText(config.get('timeframe', '15m'))
        self.leverage_input.setText(str(config.get('leverage', 50)))
        self.contract_size_input.setText(str(config.get('contract_size', 10)))
        self.take_profit_input.setText(str(config.get('take_profit_percent', 25)))
        self.stop_loss_input.setText(str(config.get('stop_loss_percent', 50)))
        self.cooldown_input.setText(str(config.get('cooldown_minutes', 15)))
        self.position_check_input.setText(str(config.get('position_check_interval', 0.1)))
        self.long_checkbox.setChecked(config.get('enable_long', True))
        self.short_checkbox.setChecked(config.get('enable_short', True))

        left_layout.addRow("API Key:", self.api_key_input)
        left_layout.addRow("Secret Key:", self.secret_key_input)
        left_layout.addRow("Password:", self.password_input)
        left_layout.addRow("Proxy HTTP:", self.proxy_http_input)
        left_layout.addRow("Proxy HTTPS:", self.proxy_https_input)
        left_layout.addRow("基础币种:", self.base_currency_input)
        left_layout.addRow("K线周期:", self.timeframe_combo)
        left_layout.addRow("杠杆倍数:", self.leverage_input)
        left_layout.addRow("合约张数:", self.contract_size_input)
        left_layout.addRow("止盈百分比 (%):", self.take_profit_input)
        left_layout.addRow("止损百分比 (%):", self.stop_loss_input)
        left_layout.addRow("冷却时间 (分钟):", self.cooldown_input)
        left_layout.addRow("仓位刷新 (秒钟):", self.position_check_input)
        left_layout.addRow(self.long_checkbox)
        left_layout.addRow(self.short_checkbox)

        self.start_button = QPushButton("▶️ 启动策略")
        self.stop_button = QPushButton("⏹ 停止策略")
        self.save_config_button = QPushButton("💾 保存配置")
        self.load_config_button = QPushButton("📂 加载配置")
        self.stop_button.setEnabled(False)

        left_layout.addRow(self.save_config_button)
        left_layout.addRow(self.load_config_button)
        left_layout.addRow(self.start_button)
        left_layout.addRow(self.stop_button)

        left_panel.setLayout(left_layout)
        main_layout.addWidget(left_panel, stretch=1)

        middle_panel = QGroupBox("实时行情 & 信号")
        middle_layout = QFormLayout()

        self.current_price_label = QLabel("--")
        self.high_price_label = QLabel("--")
        self.low_price_label = QLabel("--")
        self.upper_band_label = QLabel("--")
        self.lower_band_label = QLabel("--")
        self.signal_label = QLabel("无信号")

        self.current_price_label.setStyleSheet("font-size: 16px; color: #00ffaa;")
        self.high_price_label.setStyleSheet("font-size: 14px; color: #00ff00;")
        self.low_price_label.setStyleSheet("font-size: 14px; color: #ff3333;")
        self.upper_band_label.setStyleSheet("font-size: 14px; color: #00aaff;")
        self.lower_band_label.setStyleSheet("font-size: 14px; color: #00aaff;")
        self.signal_label.setStyleSheet("font-size: 16px; font-weight: bold; color: #ffff00;")

        middle_layout.addRow("当前开盘价:", self.current_price_label)
        middle_layout.addRow("本K线最高价:", self.high_price_label)
        middle_layout.addRow("本K线最低价:", self.low_price_label)
        middle_layout.addRow("通道上轨:", self.upper_band_label)
        middle_layout.addRow("通道下轨:", self.lower_band_label)
        middle_layout.addRow("交易信号:", self.signal_label)

        middle_panel.setLayout(middle_layout)
        main_layout.addWidget(middle_panel, stretch=1)

        right_panel = QGroupBox("持仓状态 & 日志")
        right_layout = QVBoxLayout()

        position_group = QGroupBox("持仓状态")
        pos_layout = QFormLayout()

        self.position_status = QLabel("未持仓")
        self.direction_label = QLabel("--")
        self.entry_price_label = QLabel("--")
        self.return_label = QLabel("--%")

        pos_layout.addRow("是否持仓:", self.position_status)
        pos_layout.addRow("持仓方向:", self.direction_label)
        pos_layout.addRow("入场价格:", self.entry_price_label)
        pos_layout.addRow("当前收益率:", self.return_label)

        position_group.setLayout(pos_layout)
        right_layout.addWidget(position_group)

        log_group = QGroupBox("运行日志")
        log_layout = QVBoxLayout()

        self.log_text = QTextBrowser()
        self.log_text.setReadOnly(True)
        self.log_text.setStyleSheet("""
            background-color: #1a1a2e;
            border: 1px solid #333;
            padding: 8px;
            font-size: 13px;
            color: #cccccc;
        """)

        log_layout.addWidget(self.log_text)
        log_group.setLayout(log_layout)
        right_layout.addWidget(log_group)

        right_panel.setLayout(right_layout)
        main_layout.addWidget(right_panel, stretch=2)

        self.start_button.clicked.connect(self.start_strategy)
        self.stop_button.clicked.connect(self.stop_strategy)
        self.save_config_button.clicked.connect(self.save_gui_config)
        self.load_config_button.clicked.connect(self.load_gui_config)

        self.strategy_thread = None

    def start_strategy(self) -> None:
        self.append_log("[DEBUG] 🔁 正在尝试启动策略线程...")
        try:
            position_check_interval = float(self.position_check_input.text())
            if position_check_interval < 0.01:
                self.append_log("[ERROR] ⚠️ 时间间隔不能小于 0.01 秒！")
                return
        except ValueError:
            self.append_log("[ERROR] ⚠️ 请输入合法的数字作为时间间隔！")
            return

        api_key = self.api_key_input.text()
        secret_key = self.secret_key_input.text()
        password = self.password_input.text()
        proxy_http = self.proxy_http_input.text()
        proxy_https = self.proxy_https_input.text()
        base_currency = self.base_currency_input.text().strip().upper()
        leverage = int(self.leverage_input.text())
        contract_size = float(self.contract_size_input.text())
        take_profit = int(self.take_profit_input.text())
        stop_loss = int(self.stop_loss_input.text())
        timeframe = self.timeframe_combo.currentText()
        cooldown_minutes = int(self.cooldown_input.text())
        enable_long = self.long_checkbox.isChecked()
        enable_short = self.short_checkbox.isChecked()
        proxies = {'http': proxy_http, 'https': proxy_https}

        self.strategy_thread = StrategyWorker(
            api_key=api_key,
            secret_key=secret_key,
            password=password,
            proxies=proxies,
            base_currency=base_currency,
            leverage=leverage,
            contract_size=contract_size,
            take_profit_percent=take_profit,
            stop_loss_percent=stop_loss,
            timeframe=timeframe,
            cooldown_minutes=cooldown_minutes,
            enable_long=enable_long,
            enable_short=enable_short,
            position_check_interval=position_check_interval
        )

        self.strategy_thread.log_signal.connect(self.append_log)
        self.strategy_thread.update_price_signal.connect(self.update_current_prices)
        self.strategy_thread.update_bands_signal.connect(self.update_keltner_bands)
        self.strategy_thread.update_position_signal.connect(self.update_position_status)
        self.strategy_thread.start()

        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)

    def stop_strategy(self) -> None:
        if hasattr(self, 'strategy_thread') and self.strategy_thread.isRunning():
            self.strategy_thread.stop()
            self.strategy_thread.wait()
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)

    def update_current_prices(self, current_open: float, high: float, low: float) -> None:
        self.levers = len(str(high).split('.')[-1])
        self.current_price_label.setText(f"{current_open}")
        self.high_price_label.setText(f"{high}")
        self.low_price_label.setText(f"{low}")

    def update_keltner_bands(self, upper: float, lower: float) -> None:
        self.upper_band_label.setText(f"{upper: .{self.levers}f}")
        self.lower_band_label.setText(f"{lower: .{self.levers}f}")

    def update_position_status(self, status: str, direction: str = "--", entry_price: float = 0.0,
                               profit: str = "--") -> None:
        self.position_status.setText(status)
        self.direction_label.setText(direction)
        self.entry_price_label.setText(f"{entry_price}" if entry_price != 0 else "--")
        self.return_label.setText(profit)
        if status == "已持仓":
            self.position_status.setStyleSheet("color: #00ffaa; font-weight: bold;")
        else:
            self.position_status.setStyleSheet("color: gray; font-weight: normal;")

    def append_log(self, message: str) -> None:
        self.log_text.append(message)
        self.log_text.verticalScrollBar().setValue(self.log_text.verticalScrollBar().maximum())

    def save_gui_config(self) -> None:
        config = {
            "api_key": self.api_key_input.text(),
            "secret_key": self.secret_key_input.text(),
            "password": self.password_input.text(),
            "proxies": {
                "http": self.proxy_http_input.text(),
                "https": self.proxy_https_input.text()
            },
            "base_currency": self.base_currency_input.text(),
            "leverage": int(self.leverage_input.text()),
            "contract_size": float(self.contract_size_input.text()),
            "take_profit_percent": int(self.take_profit_input.text()),
            "stop_loss_percent": int(self.stop_loss_input.text()),
            "timeframe": self.timeframe_combo.currentText(),
            "cooldown_minutes": int(self.cooldown_input.text()),
            "position_check_interval": float(self.position_check_input.text()),
            "enable_long": self.long_checkbox.isChecked(),
            "enable_short": self.short_checkbox.isChecked()
        }
        save_config(config)
        self.append_log("[INFO] ✅ 配置已保存至 config.json")

    def load_gui_config(self) -> None:
        config = load_config()
        self.api_key_input.setText(config.get('api_key', ''))
        self.secret_key_input.setText(config.get('secret_key', ''))
        self.password_input.setText(config.get('password', ''))
        self.proxy_http_input.setText(config.get('proxies', {}).get('http', 'socks5h://127.0.0.1:7890'))
        self.proxy_https_input.setText(config.get('proxies', {}).get('https', 'socks5h://127.0.0.1:7890'))
        self.base_currency_input.setText(config.get('base_currency', 'BTC'))
        self.timeframe_combo.setCurrentText(config.get('timeframe', '15m'))
        self.leverage_input.setText(str(config.get('leverage', 50)))
        self.contract_size_input.setText(str(config.get('contract_size', 10)))
        self.take_profit_input.setText(str(config.get('take_profit_percent', 25)))
        self.stop_loss_input.setText(str(config.get('stop_loss_percent', 50)))
        self.cooldown_input.setText(str(config.get('cooldown_minutes', 15)))
        self.position_check_input.setText(str(config.get('position_check_interval', 0.1)))
        self.long_checkbox.setChecked(config.get('enable_long', True))
        self.short_checkbox.setChecked(config.get('enable_short', True))
        self.append_log("[INFO] 🔄 配置已从 config.json 加载")


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = TradingStrategyGUI()
    window.showMaximized()
    sys.exit(app.exec_())
