# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.2
#   kernelspec:
#     display_name: TimeSeriesForecastingInPython
#     language: python
#     name: python3
# ---

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter, YearLocator



# %%
df = pd.read_csv('../data/jj.csv')
df['date'] = pd.to_datetime(df['date'])

df.head()

# %%
df.tail()

# %% [markdown]
# # Plot data with train/test split 

# %%
fig, ax = plt.subplots()

ax.plot(df["date"], df["data"])

ax.set_xlabel("Date")
ax.set_ylabel("Earnings per share (USD)")

start_highlight_date = pd.to_datetime("1980-01-01")
end_hightlight_date = pd.to_datetime("1980-10-01")

ax.axvspan(start_highlight_date, end_hightlight_date, color="#808080", alpha=0.2)

ax.xaxis.set_major_locator(YearLocator(2))
ax.xaxis.set_major_formatter(DateFormatter("%Y"))

fig.autofmt_xdate()
plt.tight_layout()

plt.savefig("figures/CH02_F01_peixeiro.png", dpi=300)

# %% [markdown]
# # Split to train/test 

# %%
train = df[:-4]
test = df[-4:]

# %% [markdown]
# # Predict historical mean 

# %%
historical_mean = np.mean(train['data'])
historical_mean

# %%
test.loc[:, 'pred_mean'] = historical_mean

test


# %%
def mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


# %%
mape_hist_mean = mape(test['data'], test['pred_mean'])
mape_hist_mean

# %%
fig, ax = plt.subplots()

ax.plot(train['date'], train['data'], 'g-.', label='Train')
ax.plot(test['date'], test['data'], 'b-', label='Test')
ax.plot(test['date'], test['pred_mean'], 'r--', label='Predicted')
ax.set_xlabel('Date')
ax.set_ylabel('Earnings per share (USD)')

highlight_start_date = pd.to_datetime('1980-01-01')
highlight_end_date = pd.to_datetime('1980-10-01')

ax.axvspan(highlight_start_date, highlight_end_date, color='#808080', alpha=0.2)
ax.legend(loc=2)

ax.xaxis.set_major_locator(YearLocator(2)) # Major ticks every 2 years
ax.xaxis.set_major_formatter(DateFormatter('%Y')) # Format as 'YYYY'

fig.autofmt_xdate()
plt.tight_layout()

plt.savefig('figures/CH02_F06_peixeiro.png', dpi=300)

# %% [markdown]
# # Predict last year mean 

# %%
last_year_mean = np.mean(train['data'][-4:])
last_year_mean

# %%
test.loc[:, 'pred__last_yr_mean'] = last_year_mean

test

# %%
mape_last_year_mean = mape(test['data'], test['pred__last_yr_mean'])
mape_last_year_mean

# %%
fig, ax = plt.subplots()

ax.plot(train['date'], train['data'], 'g-.', label='Train')
ax.plot(test['date'], test['data'], 'b-', label='Test')
ax.plot(test['date'], test['pred__last_yr_mean'], 'r--', label='Predicted')
ax.set_xlabel('Date')
ax.set_ylabel('Earnings per share (USD)')

ax.axvspan(highlight_start_date, highlight_end_date, color='#808080', alpha=0.2)
ax.legend(loc=2)

ax.xaxis.set_major_locator(YearLocator(2))
ax.xaxis.set_major_formatter(DateFormatter('%Y'))

fig.autofmt_xdate()
plt.tight_layout()

plt.savefig('figures/CH02_F07_peixeiro.png', dpi=300)

# %% [markdown]
# # Predict last know value 

# %%
last = train['data'].iloc[-1]
last

# %%
test.loc[:, 'pred_last'] = last

test

# %%
mape_last = mape(test['data'], test['pred_last'])
mape_last

# %%
fig, ax = plt.subplots()

ax.plot(train['date'], train['data'], 'g-.', label='Train')
ax.plot(test['date'], test['data'], 'b-', label='Test')
ax.plot(test['date'], test['pred_last'], 'r--', label='Predicted')
ax.set_xlabel('Date')
ax.set_ylabel('Earnings per share (USD)')

ax.axvspan(highlight_start_date, highlight_end_date, color='#808080', alpha=0.2)
ax.legend(loc=2)

ax.xaxis.set_major_locator(YearLocator(2))
ax.xaxis.set_major_formatter(DateFormatter('%Y'))

fig.autofmt_xdate()
plt.tight_layout()

plt.savefig('figures/CH02_F08_peixeiro.png', dpi=300)

# %% [markdown]
# # Naive seasonal forecast 

# %%
test.loc[:, 'pred_last_season'] = train['data'][-4:].values

test

# %%
mape_naive_seasonal = mape(test['data'], test['pred_last_season'])
mape_naive_seasonal

# %%
fig, ax = plt.subplots()

ax.plot(train['date'], train['data'], 'g-.', label='Train')
ax.plot(test['date'], test['data'], 'b-', label='Test')
ax.plot(test['date'], test['pred_last_season'], 'r--', label='Predicted')
ax.set_xlabel('Date')
ax.set_ylabel('Earnings per share (USD)')


ax.xaxis.set_major_locator(YearLocator(2))
ax.xaxis.set_major_formatter(DateFormatter('%Y'))
ax.legend(loc=2)

fig.autofmt_xdate()
plt.tight_layout()

plt.savefig('figures/CH02_F09_peixeiro.png', dpi=300)

# %%
fig, ax = plt.subplots()

x = ['hist_mean', 'last_year_mean', 'last', 'naive_seasonal']
y = [70.00, 15.60, 30.46, 11.56]

ax.bar(x, y, width=0.4)
ax.set_xlabel('Baselines')
ax.set_ylabel('MAPE (%)')
ax.set_ylim(0, 75)

for index, value in enumerate(y):
    plt.text(x=index, y=value + 1, s=str(value), ha='center')

plt.tight_layout()

plt.savefig('figures/CH02_F10_peixeiro.png', dpi=300)

# %%
