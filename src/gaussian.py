import os
import torch


from pytorch_lightning import loggers, callbacks

import os
import tempfile
from functools import partial
import logging

logging.basicConfig(level=logging.INFO)

import matplotlib.pyplot as plt
import pyro.distributions as dist
import pyro.distributions.transforms as T
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
import seaborn as sns

import data
import wandb
from configs.parse import parse_args
import math

USE_WANDB = (
    not ("WANDB_DISABLED" in os.environ)
    or not os.environ["WANDB_DISABLED"].lower() == "true"
)


class NormalizingFlow(pl.LightningModule):
    def __init__(self, d=1):
        super().__init__()

        self.base_dist = dist.Normal(torch.zeros(d), torch.ones(d))
        self.spline_transform = T.spline_coupling(d)
        self.flow_dist = dist.TransformedDistribution(
            self.base_dist, [self.spline_transform]
        )

    def configure_optimizers(self):
        return torch.optim.Adam(self.spline_transform.parameters())

    def step(self, batch, batch_idx):
        (x_marginal, x_posterior) = batch
        log_prob = torch.logsumexp(
            self.flow_dist.log_prob(x_posterior.reshape(-1, 1)).view(x_posterior.shape),
            dim=-1,
        )

        log_prob = log_prob - math.log(x_posterior.shape[-1])

        return -log_prob.mean()

    def training_step(self, batch, batch_idx):
        return self.step(batch, batch_idx)

    def validation_step(self, batch, batch_idx):
        loss = self.step(batch, batch_idx)
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        loss = self.step(batch, batch_idx)
        return loss


def evaluate(dir, model, test_dataset, flow_samples=1024):
    num_samples = min(flow_samples, len(test_dataset))
    X_test = torch.stack([test_dataset[i][0] for i in range(num_samples)])
    X_flow = model.flow_dist.sample((flow_samples,)).numpy()

    n, d = X_flow.shape

    #     d = 1
    if d > 1:

        plt.title(r"Joint Distribution")
        plt.xlabel(r"$x_1$")
        plt.ylabel(r"$x_2$")

        plt.scatter(X_test[:, 0], X_test[:, 1], label="data", alpha=0.5)
        plt.scatter(
            X_flow[:, 0],
            X_flow[:, 1],
            color="firebrick",
            label="flow",
            alpha=0.5,
        )

        plt.legend()
        save_dir = os.path.join(dir, "joint_distribution.png")
        plt.tight_layout()
        plt.savefig(save_dir, bbox_inches="tight")
        plt.close()
        logging.info(f"Saved joint distribution to {save_dir}")

    else:
        save_dir = os.path.join(dir, "marginal_distribution.png")
        sns.kdeplot(X_test[:, 0], label="data")
        sns.kdeplot(X_flow[:, 0], label="flow")
        plt.title(r"$p(x)$")
        plt.legend()
        plt.tight_layout()
        plt.savefig(save_dir, bbox_inches="tight")
        plt.close()
        logging.info(f"Saved marginal distribution to {save_dir}")

    #         plt.subplot(1, 2, 1)
    #         sns.kdeplot(
    #             X[:, 0],
    #             label="data",
    #         )
    #         sns.kdeplot(
    #             X_flow[:, 0],
    #             label="flow",
    #         )
    #         plt.title(r"$p(x_1)$")
    #         plt.subplot(1, 2, 2)
    #         sns.kdeplot(
    #             X[:, 1],
    #             label="data",
    #         )
    #         sns.kdeplot(
    #             X_flow[:, 1],
    #             label="flow",
    #         )
    #         plt.title(r"$p(x_2)$")
    #         plt.savefig(f"marginals.png")
    #         plt.close()

    #         # sns.pairplot(pd.DataFrame(X[:, :MAX_CORNERS]), kind="kde", corner=True, plot_kws=dict(fill=True))
    #         # sns.pairplot(pd.DataFrame(X[:, :MAX_CORNERS]), corner=True, diag_kind="kde")
    #         # plt.savefig(f"pairplot_{k}_gt.png")
    #         # plt.close()
    #         # sns.pairplot(pd.DataFrame(X_flow[:, :MAX_CORNERS]), kind="kde", corner=True, plot_kws=dict(fill=True))
    #         # sns.pairplot(pd.DataFrame(X_flow[:, :MAX_CORNERS]), corner=True, diag_kind="kde")
    #         # plt.savefig(f"pairplot_{k}_flow.png")
    #         # plt.close()
    #         df_gt = pd.DataFrame(X[:, :MAX_CORNERS])
    #         df_gt["source"] = "gt"

    #         df_flow = pd.DataFrame(X_flow[:, :MAX_CORNERS])
    #         df_flow["source"] = "flow"

    #         df = pd.concat([df_gt, df_flow], ignore_index=True)

    #         sns.pairplot(df, hue="source", kind="kde", corner=True)
    #         plt.savefig(f"pairplot.png")
    #         plt.close()

    #         sns.pairplot(
    #             df,
    #             hue="source",
    #             kind="kde",
    #             corner=True,
    #             plot_kws=dict(fill=True, alpha=0.5),
    #         )
    #         plt.savefig(f"pairplot_fill.png")
    #         plt.close()

    #         sns.pairplot(
    #             df,
    #             hue="source",
    #             diag_kind="kde",
    #             corner=True,
    #             plot_kws=dict(alpha=0.5, s=6),
    #         )
    #         plt.savefig(f"pairplot_fill_diag.png")
    #         plt.close()

    #     else:


def main(args):
    dataset = partial(getattr(data, args["dataset"].pop("object")), **args["dataset"])
    train_dataset = dataset(split="train")
    valid_dataset = dataset(split="valid")
    test_dataset = dataset(split="test")

    train_loader = DataLoader(
        train_dataset, batch_size=args["batch_size"], shuffle=True
    )
    valid_loader = DataLoader(
        valid_dataset, batch_size=args["batch_size"], shuffle=False
    )
    test_loader = DataLoader(test_dataset, batch_size=args["batch_size"], shuffle=False)

    model = NormalizingFlow()
    trainer = pl.Trainer(
        **args["trainer"],
        callbacks=[callbacks.EarlyStopping(monitor="val_loss", mode="min")],
        logger=loggers.CSVLogger(args["dir"]),
        deterministic=True,
    )
    if args["train"]:
        trainer.fit(model, train_loader, valid_loader)
    trainer.test(model, test_loader)

    torch.set_grad_enabled(False)
    evaluate(dir=args["dir"], model=model, test_dataset=test_dataset)


if __name__ == "__main__":

    args = parse_args()

    exception = None
    with tempfile.TemporaryDirectory() as tmpdir:
        args["dir"] = tmpdir
        if USE_WANDB:
            wandb.init(config=args)
        try:
            main(args)
        except (Exception, KeyboardInterrupt) as e:
            exception = e

        if USE_WANDB:
            wandb.finish()
            if os.path.exists(tmpdir):
                os.system(
                    f"wandb sync {tmpdir} --clean --clean-old-hours 0 --clean-force"
                )
        else:
            if exception is None:
                os.system(f"mv {tmpdir} ../local_runs/")

    if exception is None:
        logging.info("Run finished!")
    else:
        raise (exception)
