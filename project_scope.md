# EventOracle – Project Scope

## 1. Objective
The objective of this project is to design an event-driven prediction system that identifies market-moving events from news, macroeconomic indicators, and institutional flows, and predicts their impact on key Indian financial assets such as NIFTY and BANKNIFTY. The system aims to generate actionable buy/sell signals that outperform a traditional buy-and-hold strategy.

## 2. Assets Covered
The following assets have been selected:

- NIFTY 50 – Represents the overall Indian equity market  
- BANKNIFTY – Highly sensitive to interest rate and policy changes  
- NIFTY IT – Influenced by global (especially US) economic conditions  
- NIFTY Metal – Sensitive to commodity prices and geopolitical events  
- INR/USD – Reflects currency fluctuations driven by macroeconomic factors  
- Brent Oil – Global benchmark for crude oil affecting inflation and markets  

### Selection Justification
Assets were selected based on:
- High liquidity ensuring reliable price discovery  
- Strong sensitivity to macroeconomic and geopolitical events  
- Sectoral diversity to capture varied market reactions  
- Relevance to Indian and global economic conditions  

## 3. Event Categories

### Monetary Policy Events
- Federal Reserve (FOMC decisions)
- RBI Monetary Policy (Repo rate changes)

### Inflation Indicators
- CPI (India & US)

### Economic Indicators
- GDP growth
- IIP (Index of Industrial Production)

### Institutional Flows
- FII/DII net buying and selling activity

### Commodity Events
- Brent Oil price fluctuations
- OPEC production decisions

### Geopolitical Events
- Wars, elections, trade conflicts (via GDELT/news sources)

### Regulatory Events
- SEBI policy changes

## 4. Event Representation
Each event will include:
- Timestamp  
- Event type  
- Actual vs Expected values (for macro data)  
- Event importance (High/Medium/Low)  
- Sentiment score (from news/social media)  

## 5. Assumptions
- Market reacts to events within a 1–3 day window  
- Different sectors react differently to the same event  
- Institutional flows amplify price movements  
- News sentiment reflects market expectations  

## 6. Limitations
- Noise in news and social media sentiment  
- Delayed or muted market reactions  
- Data inconsistencies across sources  
- Difficulty in quantifying geopolitical uncertainty  

## 7. Expected Outcome
The defined asset-event universe will serve as the foundation for:
- Feature engineering  
- Model training  
- Event-driven signal generation  

This structured scope ensures consistency across the data pipeline and modeling stages.
