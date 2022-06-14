import torch
from torch.utils.data import TensorDataset, DataLoader, random_split
import pyro.distributions as dist
import pyro.distributions.transforms as T
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


import configs
from load_data import load_data

# Always use same seed for data splitting.
DATA_SEED = 0
MAX_CORNERS = 2


def get_k_folds(data, k):
    """
    Split data into k folds.
    """
    n = len(data)
    valid_size = int(n / k)
    indices = torch.randperm(n, generator=torch.Generator().manual_seed(DATA_SEED))
    folds = []
    for i in range(k):
        folds.append(
            (
                indices[i * valid_size : (i + 1) * valid_size],
                torch.cat([indices[: i * valid_size], indices[(i + 1) * valid_size :]]),
            )
        )
    return folds


def logmeanexp(x, dim, keepdim=False):
    return torch.logsumexp(x, dim=dim, keepdim=keepdim) - torch.log(
        torch.tensor(x.shape[dim])
    )


def main(cfg):
    data = load_data(cfg)
    N, d = data.shape
    folds = get_k_folds(data, 5)

    best_test_losses = []

    for k, fold in enumerate(folds):
        print(f"Starting fold {k}")
        test_indices, train_indices = fold
        train_dataset = TensorDataset(data[train_indices])
        test_dataset = TensorDataset(data[test_indices])

        train_loader = DataLoader(
            train_dataset, batch_size=cfg.batch_size, shuffle=True
        )
        test_loader = DataLoader(test_dataset, batch_size=cfg.batch_size, shuffle=False)

        base_dist = dist.Normal(torch.zeros(1), torch.ones(1))
        spline_transform = T.spline_coupling(1, count_bins=16)
        flow_dist = dist.TransformedDistribution(base_dist, [spline_transform])

        optimizer = torch.optim.Adam(spline_transform.parameters(), lr=cfg.lr)

        best_average_test_loss = float("inf")

        for epoch in range(1, cfg.epochs + 1):
            train_loss = 0
            for i, (batch,) in enumerate(train_loader):
                optimizer.zero_grad()
                log_prob = flow_dist.log_prob(batch.view(-1, 1)).view(batch.shape)
                loss = -logmeanexp(log_prob, dim=0)
                loss = loss.mean()
                loss.backward()
                optimizer.step()
                flow_dist.clear_cache()

                if i % 100 == 0:
                    print(f"Epoch {epoch} batch {i} loss {loss.item():.4f}")

                train_loss += loss.item()

            average_train_loss = train_loss / len(train_loader)
            print(f"Epoch {epoch} train loss {average_train_loss:.4f}")

            test_loss = 0
            for i, (batch,) in enumerate(test_loader):
                log_prob = flow_dist.log_prob(batch.view(-1, 1)).view(batch.shape)
                loss = -logmeanexp(log_prob, dim=0)
                loss = loss.mean()
                test_loss += loss.item()

            average_test_loss = test_loss / len(test_loader)

            print(f"Epoch {epoch} test loss {average_test_loss:.4f}")

            if average_test_loss < best_average_test_loss:
                best_average_test_loss = average_test_loss
                torch.save(spline_transform.state_dict(), f"best_model_{k}.pt")

        print(f"Best test loss {best_average_test_loss:.4f}")
        best_test_losses.append(best_average_test_loss)

        # Load best model for this fold.
        spline_transform.load_state_dict(torch.load(f"best_model_{k}.pt"))

        with torch.no_grad():

            (X,) = train_dataset[:1000]
            X = X.numpy()
            X_flow = flow_dist.sample((1000,)).numpy()

            d = 1
            if d > 1:

                plt.title(r"Joint Distribution")
                plt.xlabel(r"$x_1$")
                plt.ylabel(r"$x_2$")
                plt.scatter(X[:, 0], X[:, 1], label="data", alpha=0.5)
                plt.scatter(
                    X_flow[:, 0], X_flow[:, 1], color="firebrick", label="flow", alpha=0.5
                )
                plt.legend()
                plt.savefig(f"joint_dist_{k}.png")
                plt.close()

                plt.subplot(1, 2, 1)
                sns.kdeplot(
                    X[:, 0],
                    label="data",
                )
                sns.kdeplot(
                    X_flow[:, 0],
                    label="flow",
                )
                plt.title(r"$p(x_1)$")
                plt.subplot(1, 2, 2)
                sns.kdeplot(
                    X[:, 1],
                    label="data",
                )
                sns.kdeplot(
                    X_flow[:, 1],
                    label="flow",
                )
                plt.title(r"$p(x_2)$")
                plt.savefig(f"marginals_{k}.png")
                plt.close()

                # sns.pairplot(pd.DataFrame(X[:, :MAX_CORNERS]), kind="kde", corner=True, plot_kws=dict(fill=True))
                # sns.pairplot(pd.DataFrame(X[:, :MAX_CORNERS]), corner=True, diag_kind="kde")
                # plt.savefig(f"pairplot_{k}_gt.png")
                # plt.close()
                # sns.pairplot(pd.DataFrame(X_flow[:, :MAX_CORNERS]), kind="kde", corner=True, plot_kws=dict(fill=True))
                # sns.pairplot(pd.DataFrame(X_flow[:, :MAX_CORNERS]), corner=True, diag_kind="kde")
                # plt.savefig(f"pairplot_{k}_flow.png")
                # plt.close()
                df_gt = pd.DataFrame(X[:, :MAX_CORNERS])
                df_gt["source"] = "gt"

                df_flow = pd.DataFrame(X_flow[:, :MAX_CORNERS])
                df_flow["source"] = "flow"

                df = pd.concat([df_gt, df_flow], ignore_index=True)

                sns.pairplot(df, hue="source", kind="kde", corner=True)
                plt.savefig(f"pairplot_{k}.png")
                plt.close()

                sns.pairplot(
                    df,
                    hue="source",
                    kind="kde",
                    corner=True,
                    plot_kws=dict(fill=True, alpha=0.5),
                )
                plt.savefig(f"pairplot_{k}_fill.png")
                plt.close()

                sns.pairplot(
                    df,
                    hue="source",
                    diag_kind="kde",
                    corner=True,
                    plot_kws=dict(alpha=0.5, s=6),
                )
                plt.savefig(f"pairplot_{k}_fill_diag.png")
                plt.close()

            else: 
                sns.kdeplot(X[:, 0], label="data")
                sns.kdeplot(X_flow[:, 0], label="flow")
                plt.title(r"$p(x)$")
                plt.legend()
                plt.savefig(f"marginals_{k}.png")
                plt.close()

    print(f"Mean best test loss {sum(best_test_losses) / len(best_test_losses):.4f}")


if __name__ == "__main__":
    cfg = configs.parse_cfg()
    main(cfg)
