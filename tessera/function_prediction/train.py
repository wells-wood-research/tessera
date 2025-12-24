import argparse
import typing as t
from pathlib import Path

import pytorch_lightning as L
import torch
import torch.nn as nn
import torch.optim as optim
import torchmetrics
from lightning.pytorch.loggers import MLFlowLogger
from pytorch_lightning import LightningModule
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from torch_geometric.loader import DataLoader

from tessera.fragments.fragments_classifier import EnsembleFragmentClassifier
from tessera.function_prediction.graph_dataset import (
    GraphCreator,
    GraphDataset,
    create_graphs,
)
from tessera.function_prediction.model import FragmentNetGNN
from tessera.function_prediction.train_utils import encode_labels, log_model_summary, benchmark
from tessera.function_prediction.uniprot_processing import (
    BeProfLoader,
    Ontology,
    UniprotDownloader,
    download_uniprot_files,
)
from tessera.training.data_processing.augmentation import SubgraphSampler
from tessera.training.train import calculate_feature_dims
import pickle
from collections import defaultdict
from tessera.function_prediction.uniprot_processing import FUNC_DICT


def setup_loggers_and_callbacks(
    experiment_name: str, run_name: str
) -> t.Tuple[MLFlowLogger, ModelCheckpoint, EarlyStopping]:
    """
    Initialize MLFlowLogger, ModelCheckpoint, and EarlyStopping callbacks.

    Parameters
    ----------
    experiment_name : str
        Name of the MLFlow experiment.
    run_name : str
        Name of the MLFlow run.

    Returns
    -------
    Tuple[MLFlowLogger, ModelCheckpoint, EarlyStopping]
        Initialized logger and callbacks.
    """
    mlf_logger = MLFlowLogger(
        experiment_name=experiment_name,
        run_name=run_name,
        log_model=True,
    )
    checkpoint_callback = ModelCheckpoint(
        save_weights_only=True,
        monitor="val_loss",
        mode="min",
        filename="{epoch:02d}-{val_loss:.2f}",
    )
    earlystopping_callback = EarlyStopping(
        monitor="val_loss", mode="min", patience=20, verbose=True
    )
    return mlf_logger, checkpoint_callback, earlystopping_callback


def setup_model(
    args: argparse.Namespace,
    batch_size: int,
    go_codes: t.List[str],
    feature_info: t.Dict[str, t.Any],
    ontology: Ontology,
    **model_kwargs,
) -> L.LightningModule:
    """
    Set up the GraphLevelGNN model.

    Parameters
    ----------
    args : argparse.Namespace
        Arguments namespace containing model configurations.
    batch_size : int
        Batch size for training.
    go_codes : List[str]
        List of GO codes.
    feature_info : Dict[str, Any]
        Feature information.
    ontology : Ontology
        Ontology object.
    **model_kwargs
        Additional model configurations.

    Returns
    -------
    L.LightningModule
        Initialized model.
    """
    return GraphLevelGNN(
        batch_size=batch_size,
        num_classes=len(go_codes),
        feature_info=feature_info,
        ontology=ontology,
        hierarchical_learning=args.hierarchical_learning,
        **model_kwargs,
    )


def setup_trainer(
    root_dir: Path,
    checkpoint_callback: ModelCheckpoint,
    earlystopping_callback: EarlyStopping,
    mlf_logger: MLFlowLogger,
    num_epochs: int,
) -> L.Trainer:
    """
    Set up the PyTorch Lightning trainer.

    Parameters
    ----------
    root_dir : Path
        Root directory for checkpoints.
    checkpoint_callback : ModelCheckpoint
        Callback for checkpointing.
    earlystopping_callback : EarlyStopping
        Callback for early stopping.
    mlf_logger : MLFlowLogger
        MLFlow logger.
    num_epochs : int
        Number of epochs.

    Returns
    -------
    L.Trainer
        Initialized trainer.
    """
    return L.Trainer(
        default_root_dir=root_dir,
        callbacks=[checkpoint_callback, earlystopping_callback],
        max_epochs=num_epochs,
        enable_progress_bar=True,
        precision="16-mixed" if torch.cuda.is_available() else 32,
        logger=mlf_logger,
        deterministic=True,
    )


def finalize_training(
    trainer: L.Trainer,
    model: L.LightningModule,
    checkpoint_callback: ModelCheckpoint,
    test_loader: DataLoader,
    mlf_logger: MLFlowLogger,
    ontology: Ontology,
) -> L.LightningModule:
    """
    Finalize the training by loading the best checkpoint and testing the model.

    Parameters
    ----------
    trainer : L.Trainer
        PyTorch Lightning trainer.
    model : L.LightningModule
        The model to finalize.
    checkpoint_callback : ModelCheckpoint
        Callback that stores the best checkpoint.
    test_loader : DataLoader
        Test DataLoader.
    mlf_logger : MLFlowLogger
        MLFlow logger.

    Returns
    -------
    model: L.LightningModule
        The trained and tested model.
    """
    best_model_path = checkpoint_callback.best_model_path
    model = GraphLevelGNN.load_from_checkpoint(best_model_path, ontology=ontology)
    model.eval()
    trainer.test(model, dataloaders=test_loader)
    mlf_logger.experiment.log_artifact(mlf_logger.run_id, best_model_path)
    return model


def train_go_classifier(
    args: argparse.Namespace,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    model_name: str,
    batch_size: int,
    go_codes: t.List[str],
    feature_info: t.Dict[str, t.Any],
    ontology: Ontology,
    num_epochs: int = 100,
    **model_kwargs,
) -> LightningModule:
    """
    Train a GO classifier using GraphLevelGNN.

    Parameters
    ----------
    args : argparse.Namespace
        Arguments namespace containing model configurations.
    train_loader : DataLoader
        Training DataLoader.
    val_loader : DataLoader
        Validation DataLoader.
    test_loader : DataLoader
        Test DataLoader.
    model_name : str
        Name of the model.
    batch_size : int
        Batch size for training.
    go_codes : List[str]
        List of GO codes.
    feature_info : Dict[str, Any]
        Feature information.
    ontology : Ontology
        Ontology object.
    num_epochs : int, optional
        Number of epochs (default is 100).
    **model_kwargs
        Additional model configurations.

    Returns
    -------
    Dict[str, Any]
        Trained model and associated information.
    """
    root_dir = Path("checkpoints") / f"GraphLevel{model_name}"
    root_dir.mkdir(parents=True, exist_ok=True)

    experiment_name = f"GO_{args.go_category}_nclasses_{len(go_codes)}_train_{len(train_loader.dataset)}_val_{len(val_loader.dataset)}_test_{len(test_loader.dataset)}"
    run_name = f"{model_name}_hidden_{args.c_hidden}_layers_{args.num_layers}_{args.dropout_rate}_nodeattr_{'-'.join(args.node_attributes)}_edgeattr_{'-'.join(args.edge_attributes)}"

    (
        mlf_logger,
        checkpoint_callback,
        earlystopping_callback,
    ) = setup_loggers_and_callbacks(experiment_name, run_name)

    mlf_logger.log_hyperparams(
        {
            "node_attributes": ", ".join(args.node_attributes),
            "edge_attributes": ", ".join(args.edge_attributes),
            "batch_size": batch_size,
            "num_classes": len(go_codes),
            "c_hidden": args.c_hidden,
            "num_layers": args.num_layers,
            "dropout_rate": args.dropout_rate,
            "train_ratio": args.train_ratio,
            "val_ratio": args.val_ratio,
            "num_epochs": num_epochs,
            "augment_data": args.augment_data,
            "seed": args.seed,
            "nclasses": len(go_codes),
            "train_size": len(train_loader.dataset),
            "val_size": len(val_loader.dataset),
            "test_size": len(test_loader.dataset),
        }
    )

    model = setup_model(
        args, batch_size, go_codes, feature_info, ontology, **model_kwargs
    )

    log_model_summary(model, val_loader, mlf_logger)

    trainer = setup_trainer(
        root_dir, checkpoint_callback, earlystopping_callback, mlf_logger, num_epochs
    )

    trainer.fit(model, train_loader, val_loader)

    model = finalize_training(
        trainer, model, checkpoint_callback, test_loader, mlf_logger, ontology
    )

    return model


class GraphLevelGNN(L.LightningModule):
    def __init__(
        self,
        batch_size: int,
        num_classes: int,
        feature_info: t.Dict[str, t.Any],
        ontology: Ontology,
        hierarchical_learning: bool = False,
        log_metrics_interval: int = 10,  
        **model_kwargs,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["ontology"])
        self.batch_size = batch_size
        self.ontology = ontology
        self.hierarchical_learning = hierarchical_learning
        self.log_metrics_interval = log_metrics_interval
        self.model = FragmentNetGNN(
            node_cat_dims=feature_info["node_cat_dims"],
            node_cont_dim=feature_info["node_cont_dim"],
            edge_cat_dims=feature_info["edge_cat_dims"],
            edge_cont_dim=feature_info["edge_cont_dim"],
            c_hidden=model_kwargs["c_hidden"],
            c_out=num_classes,
            num_layers=model_kwargs["num_layers"],
            dp_rate=model_kwargs["dp_rate"],
        )
        self.loss_module = nn.BCEWithLogitsLoss()

        # Initialize metrics using torchmetrics
        self.train_metrics = self._init_metrics(num_classes)
        self.val_metrics = self._init_metrics(num_classes)
        self.test_metrics = self._init_metrics(num_classes)

        # Child Matrix (hierarchical structure)
        self.CM = self.ontology.create_child_matrix().to(self.device)

        # Storage for test results
        self.test_results = []

    @staticmethod
    def _init_metrics(num_classes: int) -> torchmetrics.MetricCollection:
        return torchmetrics.MetricCollection(
            {
                "roc_auc": torchmetrics.AUROC(
                    task="multilabel", num_labels=num_classes, average="macro"
                ),
                "f1": torchmetrics.F1Score(
                    task="multilabel",
                    num_labels=num_classes,
                    threshold=0.5,
                    average="macro",
                ),
            }
        )

    def forward(self, data) -> t.Tuple[torch.Tensor, torch.Tensor]:
        logits = self.model(data)
        return logits, data.y

    def configure_optimizers(self) -> optim.Optimizer:
        return optim.AdamW(self.parameters())

    def training_step(self, batch, batch_idx) -> torch.Tensor:
        logits, target = self.forward(batch)
        logits, target = logits.to(self.device), target.to(self.device)

        if torch.sum(target) == 0:
            raise ValueError(
                f"Warning: No positive samples in target for index {batch_idx}"
            )

        loss = self.loss_module(logits, target.float())
        # Log the loss
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, batch_size=self.batch_size)

        return loss

    def validation_step(self, batch, batch_idx) -> torch.Tensor:
        logits, target = self.forward(batch)
        logits, target = logits.to(self.device), target.to(self.device)
        loss = self.loss_module(logits, target.float())

        # Conditionally update validation metrics
        if (self.current_epoch + 1) % self.log_metrics_interval == 0:
            self.val_metrics.update(logits, target.long())

        # Log the loss
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, batch_size=self.batch_size)

        return loss

    def test_step(self, batch, batch_idx) -> torch.Tensor:
        logits, target = self.forward(batch)
        logits, target = logits.to(self.device), target.to(self.device)
        loss = self.loss_module(logits, target.float())
        # Compute probabilities from logits
        probabilities = torch.sigmoid(logits)
        # Update test metrics with probabilities and targets
        self.test_metrics(probabilities, target.long())

        # Store structure names and initialize predictions dictionary
        structure_names = [graph.name for graph in batch.to_data_list()]
        predictions_dict = {}
        for i, structure_name in enumerate(structure_names):
            # Get top 20 GO terms with the highest probabilities
            top_indices = torch.topk(probabilities[i], 20).indices.tolist()
            top_terms = [(self.ontology.go_terms[j], probabilities[i, j].item()) for j in top_indices]

            # Classify GO terms into 'bp', 'cc', 'mf' categories using a defaultdict for simplicity
            categories = defaultdict(dict)

            for go_term, prob in top_terms:
                ancestors = self.ontology.get_ancestors_is_a(go_term)

                # Map GO terms to their respective categories
                for ancestor in ancestors:
                    if ancestor in FUNC_DICT.values():
                        namespace = next(key for key, value in FUNC_DICT.items() if value == ancestor)
                        categories[namespace][go_term] = prob
                        break

            # Convert defaultdict back to a regular dict for serialization
            predictions_dict[structure_name] = dict(categories)
        
        # Save the prediction data to a file (e.g., pickle)
        output_file = "predict_result.pkl" # TODO improve
        with open(output_file, "wb") as f:
            pickle.dump(predictions_dict, f)
        # Compute custom metrics using benchmark function, assuming it takes probabilities
        metrics = benchmark(probabilities, target, self.ontology)
        
        # Log each metric with on_step=False and on_epoch=True
        for metric_name, metric_value in metrics.items():
            self.log(f"test_{metric_name}", metric_value, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        
        # Also log the loss
        self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        return loss

    def on_validation_epoch_end(self) -> None:
        if (self.current_epoch + 1) % self.log_metrics_interval == 0:
            self._log_epoch_metrics(self.val_metrics, phase="val")

    def on_test_epoch_end(self) -> None:
        self._log_epoch_metrics(self.test_metrics, phase="test")
        self._save_test_results()

    def _log_epoch_metrics(
        self, metrics: torchmetrics.MetricCollection, phase: str
    ) -> None:
        """
        Log metrics at the end of the epoch.

        Parameters
        ----------
        metrics : torchmetrics.MetricCollection
            Collection of metrics to log.
        phase : str
            Phase of training ("train", "val", "test").
        """
        metric_dict = metrics.compute()
        metric_dict = {f"{phase}_{k}": v for k, v in metric_dict.items()}
        self.log_dict(metric_dict, on_epoch=True, prog_bar=True)
        metrics.reset()
        print(f"Logged {phase} metrics: {metric_dict}")

    def _save_test_results(self) -> None:
        output_file = "test_results.txt"
        with open(output_file, "w") as f:
            f.write("StructureName,Prediction,Label\n")
            for structure_name, prediction, label in self.test_results:
                f.write(f"{structure_name},{prediction},{label}\n")

        self.logger.experiment.log_artifact(self.logger.run_id, output_file)
        Path(output_file).unlink()
        print(f"Test results saved to {output_file}")

    def get_node_attention_weights(self) -> torch.Tensor:
        return self.model.get_node_attention_weights()



def main(args: argparse.Namespace):
    args.beprof_path = Path(args.beprof_path)
    args.fragment_path = Path(args.fragment_path)
    assert args.beprof_path.exists(), f"Input file {args.beprof_path} does not exist"
    assert (
        args.fragment_path.exists()
    ), f"Input file {args.fragment_path} does not exist"

    # Load data and preprocess
    beprof_loader = BeProfLoader(
        args.beprof_path, go_category=args.go_category, debug_mode=args.debug_mode
    )
    data_dict = beprof_loader.load_data()

    # Download Uniprot files
    uniprot_downloader = UniprotDownloader(args.beprof_path)
    uniprot_to_path_dict, uniprot_to_accession_dict = download_uniprot_files(
        uniprot_downloader, data_dict
    )

    # Create graphs
    classifier = EnsembleFragmentClassifier(
        args.fragment_path,
        difference_types=["angle", "angle"],
        difference_names=["LogPr", "RamRmsd"],
        n_processes=args.num_workers,
        step_size=1,
    )
    graph_creator = GraphCreator(classifier, args.beprof_path, verbose=args.verbose)
    uniprot_to_graphs, data_dict = create_graphs(
        graph_creator, uniprot_to_path_dict, data_dict
    )

    # Create Ontology and encode labels
    ontology = Ontology(args.beprof_path)
    uniprot_id_labels, one_hot_labels = encode_labels(ontology, data_dict)

    # Create Datasets and DataLoaders
    dataloaders = {}
    for data_split in data_dict.keys():
        print(f"Data Split: {data_split}")
        print(f"    Number of Uniprot IDs: {len(uniprot_id_labels[data_split])}")
        print(f"    Number of Graphs: {len(uniprot_to_graphs[data_split])}")
        print(f"    Number of Labels: {one_hot_labels[data_split].shape}")
        current_dataset = GraphDataset(
            root=args.beprof_path,
            uniprot_to_graph_paths=uniprot_to_graphs[data_split],
            one_hot_labels=one_hot_labels[data_split],
            uniprot_id_labels=uniprot_id_labels[data_split],
            node_attributes=args.node_attributes,
            edge_attributes=args.edge_attributes,
            transform=SubgraphSampler(0.8)
            if (data_split == "train" and args.augment_data)
            else None,
        )
        dataloaders[data_split] = DataLoader(
            current_dataset,
            batch_size=args.batch_size,
            shuffle=data_split == "train",
            num_workers=args.num_workers,
            persistent_workers=True,
            pin_memory=False,
        )
    # Calculate feature dimensions
    feature_info = calculate_feature_dims(current_dataset, verbose=args.verbose)

    # Train the model
    model = train_go_classifier(
        args,
        train_loader=dataloaders["train"],
        val_loader=dataloaders["valid"],
        test_loader=dataloaders["test"],
        model_name="GNNModel",
        batch_size=args.batch_size,
        go_codes=ontology.go_terms,
        feature_info=feature_info,
        c_hidden=args.c_hidden,
        num_layers=args.num_layers,
        dp_rate=args.dropout_rate,
        layer_name="GCN",
        c_out=ontology.num_classes,
        ontology=ontology,
        num_epochs=args.num_epochs,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load BeProf Dataset")
    parser.add_argument(
        "--beprof_path", type=str, required=True, help="Path to the BeProf Dataset"
    )
    parser.add_argument(
        "--go_category",
        type=str,
        choices=["cc", "mf", "bp", "all"],
        default="all",
        help="Category of data to load",
    )
    parser.add_argument(
        "--debug_mode",
        action="store_true",
        help="Enable debug mode to load only first 10 entries of each dataset",
    )
    parser.add_argument(
        "--fragment_path", type=str, required=True, help="Path to fragment classifier"
    )
    parser.add_argument(
        "--node_attributes",
        type=str,
        nargs="+",
        required=True,
        help="List of node attributes",
    )
    parser.add_argument(
        "--edge_attributes",
        type=str,
        nargs="+",
        required=True,
        help="List of edge attributes",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=9,
        help="Number of workers to use for processing",
    )
    parser.add_argument("--seed", type=int, default=42, help="Seed for reproducibility")
    parser.add_argument(
        "--batch_size", type=int, default=10, help="Batch size for training"
    )
    parser.add_argument(
        "--c_hidden", type=int, default=500, help="Size of the hidden layers"
    )
    parser.add_argument(
        "--num_layers", type=int, default=3, help="Number of layers in the model"
    )
    parser.add_argument(
        "--dropout_rate", type=float, default=0.1, help="Dropout rate for the model"
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Print additional information"
    )
    parser.add_argument(
        "--train_ratio", type=float, default=0.8, help="Training set ratio"
    )
    parser.add_argument(
        "--val_ratio", type=float, default=0.1, help="Validation set ratio"
    )
    parser.add_argument(
        "--num_epochs", type=int, default=100, help="Number of epochs to train"
    )
    parser.add_argument(
        "--augment_data", action="store_true", help="Augment data during training"
    )
    parser.add_argument(
        "--hierarchical_learning",
        type=bool,
        default=False,
        help="Enable hierarchical learning using relationships in the ontology",
    )
    args = parser.parse_args()
    main(args)
