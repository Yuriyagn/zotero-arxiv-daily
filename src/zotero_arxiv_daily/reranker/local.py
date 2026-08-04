from .base import BaseReranker, register_reranker
import logging
import os
import warnings
from pathlib import Path
import numpy as np
@register_reranker("local")
class LocalReranker(BaseReranker):
    def get_similarity_score(self, s1: list[str], s2: list[str]) -> np.ndarray:
        cache_dir = Path(self.config.reranker.local.get("cache_dir", "models/huggingface")).resolve()
        os.environ["HF_HOME"] = str(cache_dir)
        os.environ["HUGGINGFACE_HUB_CACHE"] = str(cache_dir / "hub")
        os.environ["TRANSFORMERS_CACHE"] = str(cache_dir / "transformers")
        os.environ["HF_MODULES_CACHE"] = str(cache_dir / "modules")
        os.environ["SENTENCE_TRANSFORMERS_HOME"] = str(cache_dir / "sentence-transformers")

        from sentence_transformers import SentenceTransformer
        if not self.config.executor.debug:
            from transformers.utils import logging as transformers_logging
            from huggingface_hub.utils import logging as hf_logging
    
            transformers_logging.set_verbosity_error()
            hf_logging.set_verbosity_error()
            logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
            logging.getLogger("sentence_transformers.SentenceTransformer").setLevel(logging.ERROR)
            logging.getLogger("transformers").setLevel(logging.ERROR)
            logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
            logging.getLogger("huggingface_hub.utils._http").setLevel(logging.ERROR)
            warnings.filterwarnings("ignore", category=FutureWarning)

        encoder = SentenceTransformer(
            self.config.reranker.local.model,
            trust_remote_code=True,
            cache_folder=str(cache_dir / "sentence-transformers"),
        )
        if self.config.reranker.local.encode_kwargs:
            encode_kwargs = self.config.reranker.local.encode_kwargs
        else:
            encode_kwargs = {}
        s1_feature = encoder.encode(s1,**encode_kwargs,show_progress_bar=True)
        s2_feature = encoder.encode(s2,**encode_kwargs,show_progress_bar=True)
        sim = encoder.similarity(s1_feature, s2_feature)
        return sim.numpy()
