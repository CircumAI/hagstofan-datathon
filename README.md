# Hagstofan-Datathon CircumAI

Sales Prediction Model with 90% Confidence Intervals - Interactive Web Application

## Overview

This project predicts future export services sales based on historical data with optimistic and pessimistic bounds, providing confidence that actual values lie between these boundaries.

The model uses historical export services data from Iceland (2018-2025) to:
- Train three independent non-linear models for lower bound, median, and upper bound
- Generate predictions with asymmetric, curved confidence intervals
- Provide an interactive web interface to explore different scenarios

## 🚀 Quick Start - Web Application

**Quick start with Docker Compose:**
```bash
docker-compose up -d
```

Then open your browser to `https://localhost`

### Features

The web application includes:
- **Interactive slider** to adjust training data percentage (where prediction starts)
- **Confidence level selector** (90%, 95%, or 99%)
- **Lookback window adjustment** (3-12 months)
- **Real-time visualization** with Plotly interactive charts
- **Performance metrics** displayed dynamically
- **Sample predictions table** showing detailed results

See [WEB_APP_README.md](WEB_APP_README.md) for detailed instructions

## Results

The model achieves:
- **79.5% training coverage** - predictions fit closely to training data
- **Non-linear boundaries** - each bound follows its own curved trajectory
- **Asymmetric intervals** - lower and upper bounds behave independently
- **MAPE of ~26%** - reasonable prediction accuracy

## Data

The sales data represents monthly export services from Iceland (in MISK) over approximately 90 months from 2018-2025.

## Model Details

- **Algorithm**: Three independent Gradient Boosting Quantile Regression models
- **Architecture**: 800 estimators, depth 7, learning rate 0.015
- **Non-linearity**: Tree-based ensembles create curved, non-straight boundaries
- **Quantiles**: 5th percentile (pessimistic), 50th percentile (median), 95th percentile (optimistic)
- **Features**: Each model maintains its own prediction sequence with varying step-to-step changes
