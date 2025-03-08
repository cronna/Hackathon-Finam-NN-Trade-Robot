# конфигурационный файл для торговой стратегии

class TradeConfig:
    def __init__(self):
        self.training_NN = {"BTC", "ETH", "XRP", "LTC", "BCH"}
        self.portfolio = {"BTC", "ETH", "XRP", "LTC", "BCH"}
        self.security_board = "TQBR"
        self.timeframe_0 = "M1"
        self.timeframe_1 = "M10"
        self.start = "2021-01-01"
        self.trading_hours_start = "10:00"
        self.trading_hours_end = "23:50"
        self.period_sma_slow = 64
        self.period_sma_fast = 16
        self.draw_window = 128
        self.steps_skip = 16
        self.draw_size = 128

# Создаем экземпляр конфига
trade_config = TradeConfig()
