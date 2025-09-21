import hydra
from omegaconf import DictConfig, OmegaConf

from ..eval.evaluator import EvalConfig, Evaluator


@hydra.main(config_path="../../configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    # Allow CLI overrides: data.dataset_name, data.data_root, model.backbone.checkpoint_path
    dataset_name = cfg.data.get("dataset_name", cfg.data.get("name", "synthetic"))
    data_root = cfg.data.get("data_root", cfg.data.get("root", None))
    max_episodes = cfg.data.get("max_episodes", 100)
    split = cfg.data.get("split", "val_unseen")
    language = cfg.data.get("language", "en")

    backbone_type = "hamt" if "hamt" in str(cfg.model.backbone._target_).lower() else "dummy"
    ckpt = cfg.model.backbone.get("checkpoint_path", None)
    device = cfg.device

    econfig = EvalConfig(
        backbone_type=backbone_type,
        checkpoint_path=ckpt,
        device=device,
        dataset=dataset_name if dataset_name != "rxr_fg" else "rxr_fg",
        data_path=str(data_root) if data_root else None,
        num_episodes=int(max_episodes) if max_episodes else 100,
        batch_size=1,
        use_dygrav=True,
        tau_conf=cfg.model.policy.get("use_learned_gate", False) and 0.55 or 0.55,
        tau_entropy=1.25,
        max_candidates=4,
        vlm_model=cfg.vlm.get("model_name", "ViT-L-14"),
        vlm_pretrained=cfg.vlm.get("pretrained", "openai"),
        iou_threshold=cfg.eval.get("referent_iou_tau", 0.5) if hasattr(cfg, "eval") else 0.5,
        output_dir="experiments/eval",
        verbose=True,
    )

    # Apply split/language to dataset loader via Evaluator construction
    evaluator = Evaluator(econfig)
    # Override split/language after init
    evaluator.dataset_loader.split = split
    evaluator.dataset_loader.language = language

    metrics = evaluator.run()
    out = {
        "trigger_rate": metrics.trigger_rate,
        "GA": metrics.grounding_accuracy,
        "SR": metrics.success_rate,
    }
    print(f"[eval] trigger_rate={out['trigger_rate']:.3f} GA={out['GA']:.3f} SR={out['SR']:.3f}")


if __name__ == "__main__":
    main()
