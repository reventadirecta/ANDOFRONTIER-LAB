from .analysis import run_base_analysis
from .autoencoder import run_autoencoder_analysis
from .frames import extract_frames, frame_paths
from .pca_analysis import run_pca_analysis
from .roi import save_roi_frames


def run_pipeline(config: dict, skip_autoencoder: bool = False) -> dict:
    case_id = config["case_id"]
    results = {"case_id": case_id}
    if not frame_paths(case_id):
        results["extract_frames"] = extract_frames(config)
    results["roi"] = save_roi_frames(config)
    results["base"] = run_base_analysis(config)
    results["pca"] = run_pca_analysis(config)
    if not skip_autoencoder:
        results["autoencoder"] = run_autoencoder_analysis(config)
    return results
