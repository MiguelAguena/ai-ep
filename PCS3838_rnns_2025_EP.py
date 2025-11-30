# PCS3838 - Inteligência Artificial - 2025
# Author: Marcel Rodrigues de Barros
# EP - Redes Neurais

# Objetivo: Implementar o mecanismo de codificação temporal utilizando a formulação de codificação posicional
# descrito em Attention is All you Need (Vaswani et al., 2017)
# para melhorar o desempenho ddo modelo quando há dados faltantes.
# Tanto o encoder quanto o decoder devem ser adaptados para incorporar a codificação temporal.

# Entregáveis:
# 1. Código-fonte alterado com a implementação do mecanismo de codificação temporal.
# 2. Relatório em PDF descrevendo a implementação, desafios enfrentados e resultados obtidos.
# 3. Gráficos comparativos de desempenho entre o modelo com e sem codificação temporal.

import datetime
import pathlib
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import polars as pl
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import argparse
import uniplot
import random

# --- Configuration ---
DEFAULT_PAST_LEN = 10 * 800
DEFAULT_FUTURE_LEN = 10 * 200
DEFAULT_SLIDING_WINDOW_STEP = 50
DEFAULT_BATCH_SIZE = 32
DEFAULT_HIDDEN_SIZE = 64
DEFAULT_NUM_EPOCHS = 1000
DEFAULT_LEARNING_RATE = 1e-4
DEFAULT_DATA_FILENAME = "santos_ssh.csv"
DEFAULT_TRAIN_TEST_SPLIT_DATE = "2020-06-01 00:00:00"
DEFAULT_PAST_PLOT_VIEW_SIZE = 200
DATA_REMOVAL_RATIO = 0.0

SEED = 100
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)


def load_data(file_path: pathlib.Path) -> pl.DataFrame:
    """Loads data from CSV, and sets datetime and feature types.

    Args:
        file_path (pathlib.Path): Path to the CSV file.
    Returns:
        pl.DataFrame: Loaded and preprocessed DataFrame.
    """

    df = pl.read_csv(file_path)
    df = df.with_columns(
        [
            pl.col("datetime").str.to_datetime(
                time_unit="ms",
                strict=True,
                exact=True,
                format="%Y-%m-%d %H:%M:%S+00:00",
            ),
        ]
        + [pl.col(f).cast(pl.Float32) for f in df.columns if f != "datetime"]
    )

    return df


def split_data(
    df: pl.DataFrame, split_date: datetime.datetime
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Splits the data into training and testing sets based on the split date.

    Args:
        df (pl.DataFrame): DataFrame containing the data.
        split_date (datetime.datetime): Date to split the data.
    Returns:
        tuple: Training and testing DataFrames.
    """

    train_df = df.filter(pl.col("datetime") < split_date)
    test_df = df.filter(pl.col("datetime") >= split_date)

    print(f"Train set size: {len(train_df)}")
    print(f"Test set size: {len(test_df)}")

    return train_df, test_df


def create_sequences(
    df: pl.DataFrame,
    past_len: int,
    future_len: int,
    step: int = 1,
):
    """Creates windows using a sliding window approach.

    Args:
        data (pl.DataFrame): DataFrame containing the data.
        past_len (int): Length of the past sequence in minutes.
        future_len (int): Length of the future sequence in minutes.
        step (int): Step size for the sliding window in minutes.

    Returns:
        tuple: Arrays of past and future sequences of features and timestamps
    """

    xs, xs_timestamps, xs_lengths, ys, ys_timestamps, ys_lengths = (
        [],
        [],
        [],
        [],
        [],
        [],
    )
    observer_times = np.arange(
        df["datetime"].min() + past_len, df["datetime"].max() - future_len, step
    )
    for ot in observer_times:
        lb = df["datetime"].search_sorted(ot - past_len, side="left")
        obs = df["datetime"].search_sorted(ot, side="left")
        ub = df["datetime"].search_sorted(ot + future_len, side="left")
        x = df[lb:obs].select(pl.exclude("datetime")).to_numpy()
        x_timestamps = df[lb:obs].select(pl.col("datetime")).to_numpy()
        x_length = x.shape[0]
        y = df[obs:ub].select(pl.exclude("datetime")).to_numpy()
        y_timestamps = df[obs:ub].select(pl.col("datetime")).to_numpy()
        y_length = y.shape[0]

        if x_length == 0 or y_length == 0:
            continue
        xs.append(torch.tensor(x))
        xs_timestamps.append(torch.tensor(x_timestamps))
        xs_lengths.append(torch.tensor(x_length))
        ys.append(torch.tensor(y))
        ys_timestamps.append(torch.tensor(y_timestamps))
        ys_lengths.append(torch.tensor(y_length))
    return (
        torch.nn.utils.rnn.pad_sequence(xs, batch_first=True),
        torch.nn.utils.rnn.pad_sequence(xs_timestamps, batch_first=True),
        torch.stack(xs_lengths),
    ), (
        torch.nn.utils.rnn.pad_sequence(ys, batch_first=True),
        torch.nn.utils.rnn.pad_sequence(ys_timestamps, batch_first=True),
        torch.stack(ys_lengths),
    )


def prepare_dataloaders(
    train_df_features: pl.DataFrame,
    test_df_features: pl.DataFrame,
    past_len: int,
    future_len: int,
    batch_size: int,
    sliding_window_step: int,
):
    """Creates sequences and prepares PyTorch DataLoaders.

    Args:
        train_df (pl.DataFrame): Training DataFrame.
        test_df (pl.DataFrame): Testing DataFrame.
        past_len (int): Length of the past sequence.
        future_len (int): Length of the future sequence.
        batch_size (int): Batch size for DataLoader.
        sliding_window_step (int): Step size for sliding window.

    Returns:
        tuple: Training and testing DataLoaders.
    """

    print("Creating sequences and dataloaders...")
    (x_train, x_train_timestamps, x_train_lengths), (
        y_train,
        y_train_timestamps,
        y_train_lengths,
    ) = create_sequences(
        df=train_df_features,
        past_len=past_len,
        future_len=future_len,
        step=sliding_window_step,
    )
    (x_test, x_test_timestamps, x_test_lengths), (
        y_test,
        y_test_timestamps,
        y_test_lengths,
    ) = create_sequences(
        df=test_df_features,
        past_len=past_len,
        future_len=future_len,
        step=sliding_window_step,
    )

    print(f"x_train shape: {x_train.shape}, y_train shape: {y_train.shape}")
    print(f"x_test shape: {x_test.shape}, y_test shape: {y_test.shape}")

    train_dataset = TensorDataset(
        x_train,
        x_train_timestamps,
        x_train_lengths,
        y_train,
        y_train_timestamps,
        y_train_lengths,
    )
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    test_dataset = TensorDataset(
        x_test,
        x_test_timestamps,
        x_test_lengths,
        y_test,
        y_test_timestamps,
        y_test_lengths,
    )
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return (
        train_dataloader,
        test_dataloader,
    )


# --- Model Definition ---


class ARModel(nn.Module):
    """Autoregressive RNN Model using GRU."""

    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.encoder = nn.GRU(input_size, hidden_size, batch_first=True)
        self.decoder = nn.GRU(1, hidden_size, batch_first=True)
        self.linear = nn.Linear(hidden_size, input_size)

    def encode(self, x: torch.Tensor, x_lengths: torch.Tensor):
        """Encodes the input sequence."""

        x_packed = nn.utils.rnn.pack_padded_sequence(
            x, x_lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, h_n = self.encoder(x_packed)  # h_n shape: (1, batch_size, hidden_size)
        return h_n

    def decode(self, h_n: torch.Tensor, y_lengths: torch.Tensor):
        """Decodes the sequence autoregressively."""
        batch_size = h_n.size(1)

        mock_input = torch.zeros(batch_size, y_lengths.max(), 1, device=h_n.device)
        mock_packed = nn.utils.rnn.pack_padded_sequence(
            mock_input, y_lengths.cpu(), batch_first=True, enforce_sorted=False
        )

        out, _ = self.decoder(mock_packed, h_n)

        out = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)[0]

        y_hat = self.linear(out)

        return y_hat

    def forward(self, x, x_lengths, y_lengths):
        """Forward pass: encode the past, decode the future."""
        h_n = self.encode(x, x_lengths)
        output_seq = self.decode(h_n=h_n, y_lengths=y_lengths)
        return output_seq


# --- Training and Evaluation ---


def run_train_epoch(
    model: ARModel,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
):
    model.train()

    progress_bar = tqdm(dataloader, desc="Training")

    losses = []
    # Use enumerate to get batch index for plotting
    for (
        input_features,
        input_timestamps,
        input_lengths,
        target_features,
        target_timestamps,
        target_lengths,
    ) in progress_bar:
        inputs = input_features.to(device)
        targets = target_features.to(device)
        target_seq_len = targets.shape[1]  # Get future length from target shape

        optimizer.zero_grad()

        targets = targets[:, : max(target_lengths)]

        outputs = model(inputs, input_lengths, target_lengths)  # FORWARD PASS

        loss = criterion(outputs, targets)

        loss.backward()
        optimizer.step()

        losses.append(loss.cpu().detach().item())

    return np.mean(losses)


def run_eval_epoch(
    model: ARModel,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
):
    model.eval()

    progress_bar = tqdm(dataloader, desc="Testing")

    total_loss = 0.0
    num_batches = 0
    all_contexts = []
    all_context_timestamps = []
    all_targets = []
    all_target_timestamps = []
    all_predictions = []

    for (
        input_features,
        input_timestamps,
        input_lengths,
        target_features,
        target_timestamps,
        target_lengths,
    ) in progress_bar:
        with torch.no_grad():
            inputs = input_features.to(device)
            targets = target_features.to(device)
            targets = targets[:, : max(target_lengths)]
            predictions = model(inputs, input_lengths, target_lengths)  # FORWARD PASS
            loss = criterion(predictions, targets)
            total_loss += loss.item()
            num_batches += 1
            progress_bar.set_postfix(loss=loss.item())

            inputs = [
                inputs[i, : input_lengths[i]].cpu().numpy()
                for i in range(inputs.size(0))
            ]
            input_timestamps = [
                input_timestamps[i, : input_lengths[i]].cpu().numpy()
                for i in range(input_timestamps.size(0))
            ]
            targets = [
                targets[i, : target_lengths[i]].cpu().numpy()
                for i in range(targets.size(0))
            ]
            target_timestamps = [
                target_timestamps[i, : target_lengths[i]]
                for i in range(target_timestamps.size(0))
            ]
            predictions = [
                predictions[i, : target_lengths[i]].cpu().numpy()
                for i in range(predictions.size(0))
            ]

            all_contexts += inputs
            all_context_timestamps += input_timestamps
            all_targets += targets
            all_target_timestamps += target_timestamps
            all_predictions += predictions

    avg_loss = total_loss / num_batches
    return (
        avg_loss,
        all_contexts,
        all_context_timestamps,
        all_predictions,
        all_targets,
        all_target_timestamps,
    )


def plot_results(train_losses, test_losses, epoch):
    """Plots training and testing loss curves."""
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, epoch + 1), train_losses, label="Training Loss")
    plt.plot(range(1, epoch + 1), test_losses, label="Test Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Normalized Loss (MSE)")
    plt.title("Training and Test Loss Over Epochs")
    plt.legend()
    plt.grid(True)
    plt.savefig("loss_curve.png")
    plt.show()
    plt.close()


def plot_prediction(model, X_test, y_test, device, past_len, future_len, idx=0):
    """Plots a single prediction example against the ground truth."""
    model.eval()
    with torch.no_grad():
        input_seq = X_test[idx].unsqueeze(0).to(device)  # Add batch dimension
        target_seq = y_test[idx].squeeze(-1).cpu().numpy()  # Remove channel dim

        prediction = (
            model(input_seq, future_len).squeeze(0).squeeze(-1).cpu().numpy()
        )  # Remove batch and channel dims

    input_seq_plot = (
        input_seq.squeeze(0).squeeze(-1).cpu().numpy()
    )  # Remove batch and channel dims

    plt.figure(figsize=(15, 6))
    # Plot past input
    plt.plot(range(past_len), input_seq_plot, label="Input (Past Data)", color="blue")
    # Plot ground truth future
    plt.plot(
        range(past_len, past_len + future_len),
        target_seq,
        label="Ground Truth (Future)",
        color="green",
        linestyle="--",
    )
    # Plot predicted future
    plt.plot(
        range(past_len, past_len + future_len),
        prediction,
        label="Prediction (Future)",
        color="red",
        linestyle="-.",
    )

    plt.xlabel("Time Steps")
    plt.ylabel("SSH Value")
    plt.title(f"Example Prediction vs. Ground Truth (Index {idx})")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"prediction_example_{idx}.png")
    plt.show()
    plt.close()


def main(args):
    """Main function to run the training and evaluation."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    print(f"Using device: {device}")

    # --- Data Preparation ---
    file_path = pathlib.Path(args.data_filename)
    split_date = datetime.datetime.fromisoformat(args.split_date) # Divide df de treinamento e df de teste a partir do dia 01/06/2020 (para o df de teste, é inclusivo)
    df = load_data(file_path=file_path)
    feature_names = list(df.drop("datetime").columns) # Parece que só inclui o ssh...

    train_df, test_df = split_data(df=df, split_date=split_date) #

    # --- Apply scaling ---
    # Calculate mean and std from training data only
    train_mean = train_df.select(feature_names).mean()
    train_std = train_df.select(feature_names).std()

    print(f"Scaling data using Train Mean: {train_mean}, Train Std: {train_std}")

    # Scale data
    train_data_scaled = train_df.with_columns(
        [
            (pl.col(f) - train_mean.select([f]).item()) / train_std.select([f]).item()
            for f in feature_names
        ]
    ) # Normalização
    test_data_scaled = test_df.with_columns(
        [
            (pl.col(f) - train_mean.select([f]).item()) / train_std.select([f]).item()
            for f in feature_names
        ]
    ) # Normalização

    # convert datetimes to minutes since first measurement
    train_data_scaled = train_data_scaled.with_columns(
        [
            (pl.col("datetime") - pl.col("datetime").min())
            .dt.total_minutes()
            .cast(pl.Float32)
            .alias("datetime")
        ]
    ) # Transformando em minutos

    test_data_scaled = test_data_scaled.with_columns(
        [
            (pl.col("datetime") - pl.col("datetime").min())
            .dt.total_minutes()
            .cast(pl.Float32)
            .alias("datetime")
        ]
    ) # Transformando em minutos

    # Data removal to simulate missing data
    sample_train = sorted(
        random.sample(
            range(train_data_scaled.height),
            int((1 - DATA_REMOVAL_RATIO) * train_data_scaled.height),
        )
    ) # Remover parte dos dados de treinamento
    # ALTERAR DATA_REMOVAL_RATIO DEPOIS

    train_data_scaled = train_data_scaled[sample_train]

    sample_test = sorted(
        random.sample(
            range(test_data_scaled.height),
            int((1 - DATA_REMOVAL_RATIO) * test_data_scaled.height),
        )
    ) # Remover parte dos dados de teste
    # ALTERAR DATA_REMOVAL_RATIO DEPOIS

    test_data_scaled = test_data_scaled[sample_test]

    train_dataloader, test_dataloader = prepare_dataloaders(
        train_df_features=train_data_scaled,
        test_df_features=test_data_scaled,
        past_len=int(args.past_len),
        future_len=int(args.future_len),
        batch_size=int(args.batch_size),
        sliding_window_step=int(args.sliding_window_step),
    )

    # --- Model Setup ---
    input_size = len(feature_names) # 1(?)
    model = ARModel(input_size=input_size, hidden_size=args.hidden_size).to(device) # 1(?), 64, CUDA
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    print("\n--- Starting Training ---")
    train_losses = []
    test_losses = []
    view_size = int(args.past_view_size)

    for epoch in range(1, args.num_epochs + 1):
        print(f"\nEpoch {epoch}/{args.num_epochs}")

        # PARTE IMPORTANTE

        # Training step
        train_loss = run_train_epoch(
            model=model,
            dataloader=train_dataloader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
        )
        train_losses.append(train_loss)
        print(f"Average Training Loss: {train_loss:.4f}")

        # Evaluation step
        (
            test_loss,
            contexts,
            context_timestamps,
            predictions,
            targets,
            target_timestamps,
        ) = run_eval_epoch(
            model=model,
            dataloader=test_dataloader,
            criterion=criterion,
            device=device,
        )

        window_id = len(contexts) // 2
        example_context = contexts[window_id][-view_size:]
        example_context_timestamps = context_timestamps[window_id][-view_size:]
        example_target = targets[window_id]
        example_target_timestamps = target_timestamps[window_id]
        example_prediction = predictions[window_id]

        example_context = example_context * train_std.item() + train_mean.item()
        example_target = example_target * train_std.item() + train_mean.item()
        example_prediction = example_prediction * train_std.item() + train_mean.item()

        # Plot only first feature
        example_context = example_context[:, 0]
        example_context_timestamps = example_context_timestamps[:, 0]
        example_target = example_target[:, 0]
        example_target_timestamps = example_target_timestamps[:, 0]
        example_prediction = example_prediction[:, 0]

        uniplot.plot(
            ys=[
                example_target,
                example_context,
                example_prediction,
            ],
            xs=[
                example_target_timestamps,
                example_context_timestamps,
                example_target_timestamps,
            ],
            color=True,
            legend_labels=["Target", "Context", "Prediction"],
            title=f"Epoch: {epoch}, Eval Element: {window_id}, Loss: {test_loss:.4f}",
            height=15,  # Adjust height as needed
            lines=True,  # Use lines for time series
        )

        test_losses.append(test_loss)

        print(f"Average Test Loss: {test_loss:.4f}")
        # Plot overall loss curves with uniplot
        uniplot.plot(
            ys=[train_losses, test_losses],
            xs=[np.arange(1, epoch + 1)] * 2,  # Same x-axis for both
            color=True,
            legend_labels=["Train Loss", "Test Loss"],
            title=f"Epoch: {epoch} Loss Curves",
        )

    print("\n--- Training Complete ---")

    # --- Save Model ---
    model_save_path = pathlib.Path("model_weights.pth")
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")

    # --- Results ---
    plot_results(train_losses, test_losses, args.num_epochs)

    print("Script finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Autoregressive RNN Training Script")
    parser.add_argument(
        "--past_len",
        type=int,
        default=DEFAULT_PAST_LEN,
        help="Length of past sequence input.",
    )
    parser.add_argument(
        "--future_len",
        type=int,
        default=DEFAULT_FUTURE_LEN,
        help="Length of future sequence to predict.",
    )
    parser.add_argument(
        "--sliding_window_step",
        type=int,
        default=DEFAULT_SLIDING_WINDOW_STEP,
        help="Step size for sliding window.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Batch size for training.",
    )
    parser.add_argument(
        "--hidden_size",
        type=int,
        default=DEFAULT_HIDDEN_SIZE,
        help="Number of hidden units in RNN.",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=DEFAULT_NUM_EPOCHS,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=DEFAULT_LEARNING_RATE,
        help="Learning rate for optimizer.",
    )
    parser.add_argument(
        "--data_filename",
        type=str,
        default=DEFAULT_DATA_FILENAME,
        help="Filename to save/load data.",
    )
    parser.add_argument(
        "--split_date",
        type=str,
        default=DEFAULT_TRAIN_TEST_SPLIT_DATE,
        help="Date string for train/test split.",
    )
    parser.add_argument(
        "--past_view_size",
        type=int,
        default=DEFAULT_PAST_PLOT_VIEW_SIZE,
        help="Number of past steps to show in uniplot.",
    )

    args = parser.parse_args()
    main(args)
