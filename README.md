加密货币交易系统-Crazy_Marxism_Cypto
![cs1](https://github.com/user-attachments/assets/1a623438-2b32-4663-8d52-70a23cef13d0)

策略分析：
肯特那通道策略
  1.开多
    上跟线为阳线
    本根线开盘价突破上轨
  2.开空
    上跟线为阴线
    本根线开盘价跌破下轨
  3.趋势分析
    MACD趋势分析
    ADX趋势分析
GUI：
科技风格GUI
  1.配置页面
    杠杆配置
    币种配置
    止盈止损配置
    合约张数配置
    API配置
  2.信号页面
    开盘价
    最高价
    最低价
    上轨价格
    下轨价格
  3.持仓状态
    是否持仓
    持仓方向
    入场价格
    当前收益率
  4.日志页面
    显示运行日志
    运行报错日志
    开单平单日志
    DEBUG调试日志
    

配置文件(config.json)

    {
    "api_key": "", 
    "secret_key": "",
    "password": "",
    "proxies": {
        "http": "socks5h://127.0.0.1:7890",
        "https": "socks5h://127.0.0.1:7890"
    },
    "base_currency": "XRP",
    "leverage": 50,
    "contract_size": 0.2,
    "take_profit_percent": 100,
    "stop_loss_percent": 200,
    "timeframe": "30m",
    "cooldown_minutes": 15,
    "position_check_interval": 0.1,
    "enable_long": true,
    "enable_short": true,
    "注释":{
        "api_key":"API密钥,欧意获取",
        "secret_key": "密钥,欧意获取",
        "password": "开通API设置的密码,自己记着",
        "base_currency": "交易货币,BTC,ETH,PEPE....",
        "leverage": "杠杆",
        "contract_size": "下单张数,1,10,0.1....",
        "take_profit_percent": "止盈收益率,100,50,25",
        "stop_loss_percent": "止损收益率,100,200,50",
        "timeframe": "加密货币交易时间K,15m,5m,1m,1h,30m,4h....",
        "cooldown_minutes": "冷却时间,交易系统止盈止损后的冷却周期,防止追多/空",
        "position_check_interval": "仓位刷新时间,也就是请求速度,建议0.1s/条",
        "enable_long": "是否开多单,默认true",
        "enable_short": "是否开空单,默认true"

    }


