"""
Aria Watchlist - Stock ticker list for the Aria portfolio
"""

ARIA_TICKERS = [
    # Semiconductors & Storage
    "NVDA", "AMD", "INTC", "ARM", "MU", "SNDK", "WDC",
    # Optoelectronics & Semis
    "AAOI", "AEHR", "LITE", "MRVL",
    # AI Infrastructure / Crypto Mining
    "IREN", "NBIS", "CRWV", "CIFR", "APLD", "USAR",
    # Materials & Energy
    "MP", "UUUU", "FCX", "VRT", "CEG", "OKLO",
    # Autonomy, Robotics & Defense
    "OSS", "TSLA", "PATH", "SERV", "RKLB", "ASTS",
    "PL", "SLUNR", "ONDS", "AVAV", "LMT",
]

ARIA_GROUPS = {
    "Semiconductors & Storage": ["NVDA", "AMD", "INTC", "ARM", "MU", "SNDK", "WDC"],
    "Optoelectronics & Semis": ["AAOI", "AEHR", "LITE", "MRVL"],
    "AI Infrastructure / Crypto Mining": ["IREN", "NBIS", "CRWV", "CIFR", "APLD", "USAR"],
    "Materials & Energy": ["MP", "UUUU", "FCX", "VRT", "CEG", "OKLO"],
    "Autonomy, Robotics & Defense": ["OSS", "TSLA", "PATH", "SERV", "RKLB", "ASTS", "PL", "SLUNR", "ONDS", "AVAV", "LMT"],
}
