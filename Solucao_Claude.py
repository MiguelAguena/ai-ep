# PCS3838 - Inteligência Artificial - 2025
# Author: Marcel Rodrigues de Barros
# EP - Redes Neurais - COM CODIFICAÇÃO TEMPORAL

# Objetivo: Implementar o mecanismo de codificação temporal utilizando a formulação de codificação posicional
# descrito em Attention is All you Need (Vaswani et al., 2017)
# para melhorar o desempenho do modelo quando há dados faltantes.
# Tanto o encoder quanto o decoder devem ser adaptados para incorporar a codificação temporal.

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
import math

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
DATA_REMOVAL_RATIO = 0.0  # Altere para 0.2, 0.4 para testar com dados faltantes

SEED = 100
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)


# --- Positional Encoding Module ---
class TemporalPositionalEncoding(nn.Module):
    """
    Codificação posicional temporal baseada no paper 'Attention is All You Need'.
    Adaptada para usar timestamps reais ao invés de posições sequenciais.
    """
    def __init__(self, d_model: int, max_len: int = 10000, scale_factor: float = 1.0):
        """
        Args:
            d_model: Dimensão do embedding (deve ser par)
            max_len: Comprimento máximo da sequência
            scale_factor: Fator de escala para normalizar os timestamps
        """
        super().__init__()
        self.d_model = d_model
        self.scale_factor = scale_factor
        
        # Pré-calcular as frequências para eficiência
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                            (-math.log(10000.0) / d_model))
        self.register_buffer('div_term', div_term)
    
    def forward(self, timestamps: torch.Tensor):
        """
        Args:
            timestamps: Tensor de shape (batch_size, seq_len, 1) contendo timestamps em minutos
        Returns:
            Positional encoding de shape (batch_size, seq_len, d_model)
        """
        batch_size, seq_len, _ = timestamps.shape
        
        # Normalizar timestamps pelo fator de escala
        timestamps = timestamps.squeeze(-1) / self.scale_factor  # (batch_size, seq_len)
        
        # Criar tensor para encodings
        pe = torch.zeros(batch_size, seq_len, self.d_model, device=timestamps.device)
        
        # Calcular posições com senos e cossenos
        # timestamps: (batch_size, seq_len)
        # div_term: (d_model//2,)
        # Precisamos fazer broadcasting
        timestamps_expanded = timestamps.unsqueeze(-1)  # (batch_size, seq_len, 1)
        
        # Aplicar seno nas dimensões pares
        pe[:, :, 0::2] = torch.sin(timestamps_expanded * self.div_term)
        
        # Aplicar cosseno nas dimensões ímpares
        pe[:, :, 1::2] = torch.cos(timestamps_expanded * self.div_term)
        
        return pe


def load_data(file_path: pathlib.Path) -> pl.DataFrame:
    """Loads data from CSV, and sets datetime and feature types."""
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
    """Splits the data into training and testing sets based on the split date."""
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
    """Creates windows using a sliding window approach."""
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
    """Creates sequences and prepares PyTorch DataLoaders."""
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

    return train_dataloader, test_dataloader


# --- Model Definition ---


class ARModel(nn.Module):
    """Autoregressive RNN Model using GRU with Temporal Positional Encoding."""

    def __init__(self, input_size: int, hidden_size: int, use_temporal_encoding: bool = True):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.use_temporal_encoding = use_temporal_encoding
        
        # Codificação posicional temporal
        if use_temporal_encoding:
            self.temporal_encoding = TemporalPositionalEncoding(
                d_model=hidden_size, 
                scale_factor=1000.0  # Ajuste conforme necessário
            )
            # Projeção das features de entrada para hidden_size
            self.input_projection = nn.Linear(input_size, hidden_size)
            # Encoder recebe hidden_size (features projetadas + encoding)
            self.encoder = nn.GRU(hidden_size, hidden_size, batch_first=True)
            # Decoder também usa encoding temporal
            self.decoder = nn.GRU(hidden_size, hidden_size, batch_first=True)
        else:
            # Modelo original sem encoding temporal
            self.encoder = nn.GRU(input_size, hidden_size, batch_first=True)
            self.decoder = nn.GRU(1, hidden_size, batch_first=True)
        
        self.linear = nn.Linear(hidden_size, input_size)

    def encode(self, x: torch.Tensor, x_timestamps: torch.Tensor, x_lengths: torch.Tensor):
        """Encodes the input sequence with temporal positional encoding."""
        if self.use_temporal_encoding:
            # Projetar features para hidden_size
            x_projected = self.input_projection(x)  # (batch, seq_len, hidden_size)
            
            # Adicionar encoding temporal
            temporal_enc = self.temporal_encoding(x_timestamps)
            x_encoded = x_projected + temporal_enc
        else:
            x_encoded = x
        
        x_packed = nn.utils.rnn.pack_padded_sequence(
            x_encoded, x_lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, h_n = self.encoder(x_packed)
        return h_n

    def decode(self, h_n: torch.Tensor, y_timestamps: torch.Tensor, y_lengths: torch.Tensor):
        """Decodes the sequence with temporal positional encoding."""
        batch_size = h_n.size(1)
        
        if self.use_temporal_encoding:
            # Criar encoding temporal para as posições futuras
            temporal_enc = self.temporal_encoding(y_timestamps)
            
            decoder_input_packed = nn.utils.rnn.pack_padded_sequence(
                temporal_enc, y_lengths.cpu(), batch_first=True, enforce_sorted=False
            )
        else:
            mock_input = torch.zeros(batch_size, y_lengths.max(), 1, device=h_n.device)
            decoder_input_packed = nn.utils.rnn.pack_padded_sequence(
                mock_input, y_lengths.cpu(), batch_first=True, enforce_sorted=False
            )

        out, _ = self.decoder(decoder_input_packed, h_n)
        out = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)[0]
        y_hat = self.linear(out)
        
        return y_hat

    def forward(self, x, x_timestamps, x_lengths, y_timestamps, y_lengths):
        """Forward pass: encode the past, decode the future."""
        h_n = self.encode(x, x_timestamps, x_lengths)
        output_seq = self.decode(h_n=h_n, y_timestamps=y_timestamps, y_lengths=y_lengths)
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
    
    for (
        input_features,
        input_timestamps,
        input_lengths,
        target_features,
        target_timestamps,
        target_lengths,
    ) in progress_bar:
        inputs = input_features.to(device)
        input_times = input_timestamps.to(device)
        targets = target_features.to(device)
        target_times = target_timestamps.to(device)

        optimizer.zero_grad()

        targets = targets[:, : max(target_lengths)]

        # FORWARD PASS com timestamps
        outputs = model(inputs, input_times, input_lengths, target_times, target_lengths)

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
            input_times = input_timestamps.to(device)
            targets = target_features.to(device)
            target_times = target_timestamps.to(device)
            
            targets = targets[:, : max(target_lengths)]
            
            # FORWARD PASS com timestamps
            predictions = model(inputs, input_times, input_lengths, target_times, target_lengths)
            
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


def plot_results(train_losses, test_losses, epoch, suffix=""):
    """Plots training and testing loss curves."""
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, epoch + 1), train_losses, label="Training Loss")
    plt.plot(range(1, epoch + 1), test_losses, label="Test Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Normalized Loss (MSE)")
    plt.title("Training and Test Loss Over Epochs")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"loss_curve{suffix}.png")
    plt.show()
    plt.close()


def main(args):
    """Main function to run the training and evaluation."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Temporal Encoding: {'ENABLED' if args.use_temporal_encoding else 'DISABLED'}")
    print(f"Data Removal Ratio: {DATA_REMOVAL_RATIO}")

    # --- Data Preparation ---
    file_path = pathlib.Path(args.data_filename)
    split_date = datetime.datetime.fromisoformat(args.split_date)
    df = load_data(file_path=file_path)
    feature_names = list(df.drop("datetime").columns)

    train_df, test_df = split_data(df=df, split_date=split_date)

    # --- Apply scaling ---
    train_mean = train_df.select(feature_names).mean()
    train_std = train_df.select(feature_names).std()

    print(f"Scaling data using Train Mean: {train_mean}, Train Std: {train_std}")

    train_data_scaled = train_df.with_columns(
        [
            (pl.col(f) - train_mean.select([f]).item()) / train_std.select([f]).item()
            for f in feature_names
        ]
    )
    test_data_scaled = test_df.with_columns(
        [
            (pl.col(f) - train_mean.select([f]).item()) / train_std.select([f]).item()
            for f in feature_names
        ]
    )

    # convert datetimes to minutes since first measurement
    train_data_scaled = train_data_scaled.with_columns(
        [
            (pl.col("datetime") - pl.col("datetime").min())
            .dt.total_minutes()
            .cast(pl.Float32)
            .alias("datetime")
        ]
    )
    test_data_scaled = test_data_scaled.with_columns(
        [
            (pl.col("datetime") - pl.col("datetime").min())
            .dt.total_minutes()
            .cast(pl.Float32)
            .alias("datetime")
        ]
    )

    # Data removal to simulate missing data
    sample_train = sorted(
        random.sample(
            range(train_data_scaled.height),
            int((1 - DATA_REMOVAL_RATIO) * train_data_scaled.height),
        )
    )
    train_data_scaled = train_data_scaled[sample_train]

    sample_test = sorted(
        random.sample(
            range(test_data_scaled.height),
            int((1 - DATA_REMOVAL_RATIO) * test_data_scaled.height),
        )
    )
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
    input_size = len(feature_names)
    model = ARModel(
        input_size=input_size, 
        hidden_size=args.hidden_size,
        use_temporal_encoding=args.use_temporal_encoding
    ).to(device)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    print("\n--- Starting Training ---")
    train_losses = []
    test_losses = []
    view_size = int(args.past_view_size)

    for epoch in range(1, args.num_epochs + 1):
        print(f"\nEpoch {epoch}/{args.num_epochs}")

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
            height=15,
            lines=True,
        )

        test_losses.append(test_loss)

        print(f"Average Test Loss: {test_loss:.4f}")
        uniplot.plot(
            ys=[train_losses, test_losses],
            xs=[np.arange(1, epoch + 1)] * 2,
            color=True,
            legend_labels=["Train Loss", "Test Loss"],
            title=f"Epoch: {epoch} Loss Curves",
        )

    print("\n--- Training Complete ---")

    # --- Save Model ---
    suffix = "_temporal" if args.use_temporal_encoding else "_baseline"
    model_save_path = pathlib.Path(f"model_weights{suffix}.pth")
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")

    # --- Results ---
    plot_results(train_losses, test_losses, args.num_epochs, suffix)

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
    parser.add_argument(
        "--use_temporal_encoding",
        action="store_true",
        default=True,
        help="Use temporal positional encoding (default: True).",
    )
    parser.add_argument(
        "--no_temporal_encoding",
        dest="use_temporal_encoding",
        action="store_false",
        help="Disable temporal positional encoding.",
    )

    args = parser.parse_args()
    main(args)
