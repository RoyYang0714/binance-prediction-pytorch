"""Visualization."""
import matplotlib.pyplot as plt
import seaborn as sns


def plot_price(file_name, data):
    """Plot financial price overtime."""
    sns.set_style("darkgrid")
    plt.figure(figsize=(15, 9))
    plt.plot(data[["Close"]])
    plt.xticks(
        range(0, data.shape[0], 1000), data["Open time"].loc[::1000], rotation=45
    )
    plt.title("Price", fontsize=18, fontweight="bold")
    plt.xlabel("Date", fontsize=18)
    plt.ylabel("Close Price (USD)", fontsize=18)
    plt.savefig(f"images/{file_name}_data.png")


def plot_prediction(df, ticker):
    """plot prediction results."""
    plt.figure(figsize=(30, 7))
    plt.title("Predictive model " + str(ticker))
    plt.plot(df["Open time"], df["Close"], label="Close", alpha=0.2)

    buy_df = df["Buy"] * df["Close"]

    plt.scatter(
        df["Open time"].loc[~(df.Buy == 0)],
        buy_df.loc[~(buy_df == 0)],
        label="Buy",
        marker="^",
        color="magenta",
        alpha=0.15,
    )

    sell_df = df["Sell"] * df["Close"]

    plt.scatter(
        df["Open time"].loc[~(df.Sell == 0)],
        sell_df.loc[~(sell_df == 0)],
        label="Sell",
        marker="*",
        color="blue",
        alpha=0.15,
    )

    plt.legend()
    plt.savefig(f"images/{ticker}_prediction.png")
