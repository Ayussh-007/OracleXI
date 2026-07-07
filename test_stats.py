from core.prediction_engine import PredictionEngine
import logging

logging.basicConfig(level=logging.INFO)
engine = PredictionEngine()
engine.initialize()

print("--- Testing Football ---")
stats_fb = engine.stats_analyzer.analyze_football("Argentina", "France", engine.data_engine)
print(stats_fb)

print("--- Testing Cricket ---")
stats_cr = engine.stats_analyzer.analyze_cricket("India", "Pakistan", engine.data_engine)
print(stats_cr)
