# Sem-8-final-project
# This is my final year project 
# Here my aim is to make a model that will work on the principle of volume divergence
# The aim of the model will be to predict the next high or low
# Volume Divergence is a basic but effective way to know the intrest of the investors
# In simple terms in volume divergence we check the changes in price when their is change in the average volume
# What is volume?
# Volume is just a measure of the transaction in that particular shares, let say that there is 1 share was sold and one was bought then the volume corresponding to this transaction is 1. Volunme is the count of unique transaction of shares, if there is transaction of 1M share means 1M shares are sold and bought.
# Their are four thing to keep in mind 
# 1. Positive confirmed divergence
# 2. Negative confirmed divergence
# 3. Negative fake divergence
# 4. Positive fake divergence
# Dataset preparation
# for dataset we will be using the trading view api or NSE (National Stock exchange dataset regarding the volume and the share)
# I will be using the 30 min time frame this is solely based on my intuition
# I am thinking to take volume candlestick pattern and support resistance be the input and to predict the next 3 candle
# any candle has four thing
# 1. Opening high
# 2. best high
# 3. least low
# 4. ending low
# depends on red green
# Dataset will consist of top 20 comanies of nifty50 index (india)
# With 30 min interval