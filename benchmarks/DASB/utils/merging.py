import os
import logging
from typing import List
import torch
import torch.nn as nn
from torch.func import functional_call
from audiocraft.solvers import FlexiCodecSolver

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -------------------------
# 1. The LLM Adapter 
# -------------------------
class AudioAdapter(nn.Module):
    def __init__(self, audio_dim: int, llm_dim: int, hidden_dim: int = 2048, num_layers: int = 3):
        super().__init__()
        dims = [audio_dim] + [hidden_dim] * (num_layers - 1) + [llm_dim]
        if llm_dim is None:
            llm_dim = hidden_dim
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            if i < len(dims) - 2:
                layers.append(nn.GELU())
        self.proj = nn.Sequential(*layers)
        
    def forward(self, audio_features: torch.Tensor) -> torch.Tensor:
        return self.proj(audio_features)

# -------------------------
# 2. The Base Extractor Wrapper
# -------------------------
class FlexiExtractorWrapper(nn.Module):
    def __init__(self, init_model_ckpt, num_ac_layers=0):
        super().__init__()
        self.model = FlexiCodecSolver.model_from_checkpoint(init_model_ckpt, device='cuda')
        self.num_ac_layers = num_ac_layers

    def forward(self, audio_raw):
        emb = self.model.encoder(audio_raw).permute(0, 2, 1)
        z_ac = self.model.acoustic_proj_in(emb).permute(0, 2, 1)
        z_sem = self.model.sem_proj_in(emb).permute(0, 2, 1)
        
        q_res_sem_codes = self.model.quantizer_sem.encode(z_sem)
        q_res_ac_codes = self.model.quantizer.encode(z_ac)
        
        z_sem_dec = self.model.quantizer_sem.decode(q_res_sem_codes)
        z_sem_out = self.model.sem_proj_out(z_sem_dec.permute(0, 2, 1)).permute(0, 2, 1)
        
        if self.num_ac_layers > 0:
            truncated_ac_codes = q_res_ac_codes[:, :self.num_ac_layers, :]
            z_ac_dec = self.model.quantizer.decode(truncated_ac_codes)
            z_ac_out = self.model.acoustic_proj_out(z_ac_dec.permute(0, 2, 1)).permute(0, 2, 1)
            z_combined = z_sem_out + z_ac_out
        else:
            z_combined = z_sem_out
            
        return z_combined.transpose(1, 2)

# -------------------------
# 3. The Online Merging Engine
# -------------------------
class OnlineMergingAdapter(nn.Module):
    def __init__(self, init_model_ckpt, ckpt_paths: list, merge_prefixes: list, 
                 num_ac_layers: int = 0, merge_method: str = "weighted", tsv_rank: int = None):
        """
        merge_method: 'weighted' for standard interpolation, 'tsv' for Task-Specific Vector merging
        tsv_rank: Optional rank constraint for TSV. If None, it dynamically calculates r = min(m, n) // K
        """
        super().__init__()
        
        self.merge_method = merge_method.lower()
        assert self.merge_method in ["weighted", "tsv"], "merge_method must be 'weighted' or 'tsv'"
        
        # Assume FlexiExtractorWrapper is defined elsewhere
        self.extractor = FlexiExtractorWrapper(init_model_ckpt, num_ac_layers)
        for p in self.extractor.parameters():
            p.requires_grad = False
            
        self.num_tasks = len(ckpt_paths)
        self.task_weights = nn.Parameter(torch.ones(self.num_tasks))
        self.tsv_rank = tsv_rank
        
        base_sd = self.extractor.state_dict()
        self.target_keys = []
        for key in base_sd.keys():
            check_key = key.replace('model.', '', 1) 
            if any(check_key.startswith(p) for p in merge_prefixes):
                self.target_keys.append(key)
                
        logger.info(f"Dynamically replacing {len(self.target_keys)} layers using {self.num_tasks} checkpoints.")
        logger.info(f"Using merge method: {self.merge_method.upper()}")

        finetuned_sds = [torch.load(p, map_location='cpu', weights_only=False)['best_state']['model'] for p in ckpt_paths]
            
        self.finetuned_buffers = {}
        for key in self.target_keys:
            orig_key = key.replace('model.', '', 1)
            safe_buffer_name = key.replace('.', '_')
            base_tensor = base_sd[key]
            finetuned_tensors = [sd[orig_key] for sd in finetuned_sds]
            
            if self.merge_method == "weighted":
                stacked_weights = torch.stack(finetuned_tensors, dim=0)
            elif self.merge_method == "tsv":
                # Precompute the orthogonalized task vectors (deltas)
                stacked_weights = self._compute_tsv_deltas(base_tensor, finetuned_tensors, self.tsv_rank)
                # Store the base weights since TSV requires: W_base + sum(alpha * Delta_ortho)
                self.register_buffer(f"{safe_buffer_name}_base", base_tensor.detach().clone())
            
            self.register_parameter(safe_buffer_name, nn.Parameter(stacked_weights))
            self.finetuned_buffers[key] = safe_buffer_name

        # --- VERIFICATION STEP 1: The Logging Ledger ---
        logger.info("\n--- WEIGHT MAPPING LEDGER ---")
        for key in self.target_keys:
            orig_key = key.replace('model.', '', 1)
            buffer_name = self.finetuned_buffers[key]
            stacked_w = getattr(self, buffer_name)
            base_shape = base_sd[key].shape
            
            logger.info(f"Target Module Key : '{key}'")
            logger.info(f"Loaded Ckpt Key   : '{orig_key}'")
            logger.info(f"Stacked Shape     : {stacked_w.shape} -> Expected Base: {base_shape}")
            
            assert stacked_w.shape[1:] == base_shape, \
                f"SHAPE MISMATCH! Base: {base_shape}, Ckpt: {stacked_w.shape[1:]}"
        logger.info("-----------------------------\n")

    @staticmethod
    def _as_matrix(x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 0: return x.view(1, 1)
        if x.ndim == 1: return x.view(x.shape[0], 1)
        return x.view(x.shape[0], -1)

    @staticmethod
    def _polar_orthonormalize(X: torch.Tensor) -> torch.Tensor:
        U, _, Vh = torch.linalg.svd(X, full_matrices=False)
        return U @ Vh

    def _compute_tsv_deltas(self, base_tensor: torch.Tensor, finetuned_tensors: list, per_task_rank: int):
        """Precomputes the orthogonalized task vectors for the TSV method."""
        base_mat = self._as_matrix(base_tensor)
        deltas = [self._as_matrix(t.cpu()) - base_mat.cpu() for t in finetuned_tensors]
        
        m, n = deltas[0].shape
        K = len(deltas)
        r = max(1, min(m, n) // K) if per_task_rank is None else max(1, min(per_task_rank, min(m, n)))

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        U_list, V_list, S_list = [], [], []
        
        # Iterative GPU SVD
        for D in deltas:
            D_gpu = D.to(device)
            U, S, Vh = torch.linalg.svd(D_gpu, full_matrices=False)
            U_list.append(U[:, :r].cpu())
            V_list.append(Vh.transpose(0, 1)[:, :r].cpu())
            S_list.append(S[:r].cpu())
            del D_gpu, U, S, Vh
            if device.type == 'cuda': torch.cuda.empty_cache()

        U_cat = torch.cat(U_list, dim=1).to(device)
        V_cat = torch.cat(V_list, dim=1).to(device)

        U_perp = self._polar_orthonormalize(U_cat)
        V_perp = self._polar_orthonormalize(V_cat)

        M_k_list = []
        for k, S_r in enumerate(S_list):
            s = k * r
            U_k = U_perp[:, s : s + r]
            V_k = V_perp[:, s : s + r]
            S_mat = torch.diag(S_r.to(device))
            
            # Reconstruct the orthogonalized delta for task k
            M_k_mat = U_k @ S_mat @ V_k.transpose(0, 1)
            M_k_list.append(M_k_mat.cpu().view_as(base_tensor))

        del U_cat, V_cat, U_perp, V_perp, S_mat, M_k_mat
        if device.type == 'cuda': torch.cuda.empty_cache()

        return torch.stack(M_k_list, dim=0)

    def get_dynamic_state_dict(self):
        alphas = torch.softmax(self.task_weights, dim=0)
        dynamic_params = dict(self.extractor.named_parameters())
        dynamic_params.update(dict(self.extractor.named_buffers()))
        
        for key, buffer_name in self.finetuned_buffers.items():
            stacked_w = getattr(self, buffer_name) 
            view_shape = [-1] + [1] * (stacked_w.ndim - 1)
            alpha_view = alphas.view(*view_shape)
            
            if self.merge_method == "weighted":
                # Standard weighted average: W = sum(alpha_k * W_k)
                mixed_w = (stacked_w * alpha_view).sum(dim=0)
            elif self.merge_method == "tsv":
                # TSV Task Vectors: W = W_base + sum(alpha_k * M_k)
                base_w = getattr(self, f"{buffer_name}_base")
                mixed_w = base_w + (stacked_w * alpha_view).sum(dim=0)
            
            # --- VERIFICATION STEP 2: Runtime Integrity ---
            assert key in dynamic_params, f"FATAL: Trying to replace '{key}', but it doesn't exist in the base model!"
            dynamic_params[key] = mixed_w
            
        return dynamic_params

    def forward(self, audio_raw):
        dynamic_params = self.get_dynamic_state_dict()
        z_combined = functional_call(self.extractor, dynamic_params, (audio_raw,))
        return z_combined
    
    def export_mixing_weights(self, output_path_dir: str, filename: str = "att.ckpt"):
        import os
        os.makedirs(output_path_dir, exist_ok=True)
        full_path = os.path.join(output_path_dir, filename)
        
        attention_probs = torch.softmax(self.task_weights, dim=0).detach().cpu()
        raw_logits = self.task_weights.detach().cpu()
        
        export_data = {
            "checkpoint_logits": raw_logits,
            "checkpoint_attention_probs": attention_probs,
            "num_checkpoints": len(self.task_weights),
            "merge_method": self.merge_method
        }
            
        torch.save(export_data, full_path)
# class OnlineMergingAdapter(nn.Module):
#     def __init__(self, init_model_ckpt, ckpt_paths: list, merge_prefixes: list, num_ac_layers: int = 0):
#         super().__init__()
        
#         self.extractor = FlexiExtractorWrapper(init_model_ckpt, num_ac_layers)
#         for p in self.extractor.parameters():
#             p.requires_grad = False
            
#         self.num_tasks = len(ckpt_paths)
#         self.task_weights = nn.Parameter(torch.ones(self.num_tasks))
        
#         base_sd = self.extractor.state_dict()
#         self.target_keys = []
#         for key in base_sd.keys():
#             check_key = key.replace('model.', '', 1) 
#             if any(check_key.startswith(p) for p in merge_prefixes):
#                 self.target_keys.append(key)
                
#         logger.info(f"Dynamically replacing {len(self.target_keys)} layers using {self.num_tasks} checkpoints.")

#         finetuned_sds = [torch.load(p, map_location='cpu', weights_only=False)['best_state']['model'] for p in ckpt_paths]
            
#         self.finetuned_buffers = {}
#         for key in self.target_keys:
#             orig_key = key.replace('model.', '', 1)
#             stacked_weights = torch.stack([sd[orig_key] for sd in finetuned_sds], dim=0)
            
#             safe_buffer_name = key.replace('.', '_')
#             self.register_parameter(safe_buffer_name, nn.Parameter(stacked_weights))
#             self.finetuned_buffers[key] = safe_buffer_name

#         # ... inside OnlineMergingAdapter __init__ ...
        
#         # --- VERIFICATION STEP 1: The Logging Ledger ---
#         logger.info("\n--- WEIGHT MAPPING LEDGER ---")
#         for key in self.target_keys:
#             orig_key = key.replace('model.', '', 1)
#             buffer_name = self.finetuned_buffers[key]
#             stacked_w = getattr(self, buffer_name)
#             base_shape = base_sd[key].shape
            
#             logger.info(f"Target Module Key : '{key}'")
#             logger.info(f"Loaded Ckpt Key   : '{orig_key}'")
#             logger.info(f"Stacked Shape     : {stacked_w.shape} -> Expected Base: {base_shape}")
            
#             # Crash immediately if the checkpoint tensor doesn't match the base model tensor
#             assert stacked_w.shape[1:] == base_shape, \
#                 f"SHAPE MISMATCH! Base: {base_shape}, Ckpt: {stacked_w.shape[1:]}"
#         logger.info("-----------------------------\n")

#     def get_dynamic_state_dict(self):
#         alphas = torch.softmax(self.task_weights, dim=0)
#         dynamic_params = dict(self.extractor.named_parameters())
#         dynamic_params.update(dict(self.extractor.named_buffers()))
        
#         for key, buffer_name in self.finetuned_buffers.items():
#             stacked_w = getattr(self, buffer_name) 
#             view_shape = [-1] + [1] * (stacked_w.ndim - 1)
#             alpha_view = alphas.view(*view_shape)
#             mixed_w = (stacked_w * alpha_view).sum(dim=0)
            
#             # --- VERIFICATION STEP 2: Runtime Integrity ---
#             assert key in dynamic_params, f"FATAL: Trying to replace '{key}', but it doesn't exist in the base model!"
            
#             dynamic_params[key] = mixed_w
            
#         return dynamic_params

#     def forward(self, audio_raw):
#         dynamic_params = self.get_dynamic_state_dict()
#         z_combined = functional_call(self.extractor, dynamic_params, (audio_raw,))
#         return z_combined
    
#     def export_mixing_weights(self, output_path_dir: str, filename: str = "att.ckpt"):
#         """
#         Saves the current checkpoint mixing weights (model attention) to a file.
#         """
#         import os
#         os.makedirs(output_path_dir, exist_ok=True)
        
            
#         full_path = os.path.join(output_path_dir, filename)
        
#         # Calculate the attention probabilities over the checkpoints
#         attention_probs = torch.softmax(self.task_weights, dim=0).detach().cpu()
#         raw_logits = self.task_weights.detach().cpu()
        
#         # Save explicitly as model/checkpoint attention
#         export_data = {
#             "checkpoint_logits": raw_logits,
#             "checkpoint_attention_probs": attention_probs,
#             "num_checkpoints": len(self.task_weights)
#         }
            
#         torch.save(export_data, full_path)

# -------------------------
# 4. End-to-End Probing Head
# -------------------------
class DownstreamProbe(nn.Module):
    def __init__(self, init_model_ckpt, ckpt_paths, merge_prefixes, num_classes, flexi_dim=1024, llm_dim=4096, num_ac_layers=0):
        super().__init__()
        self.merging_adapter = OnlineMergingAdapter(init_model_ckpt, ckpt_paths, merge_prefixes, num_ac_layers)
        self.audio_adapter = AudioAdapter(audio_dim=flexi_dim, llm_dim=llm_dim, hidden_dim=2048, num_layers=3)
        self.classifier = nn.Sequential(
            nn.Linear(llm_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes)
        )

    def forward(self, audio_raw):
        z_combined = self.merging_adapter(audio_raw)
        llm_embeddings = self.audio_adapter(z_combined)
        pooled = llm_embeddings.mean(dim=1) 
        logits = self.classifier(pooled)
        return logits

# -------------------------
# Execution Test
# -------------------------
# if __name__ == "__main__":
#     MERGE_PREFIXES = [
#         'sem_proj',
#         'quantizer_sem',
#     ]
    
#     # Just list the checkpoint identifiers! The adapter will learn the weights.
#     task_names = [
#         "wavlm",
#         "clap", 
#     ]
    
#     ckpt_fp = "/home/poonehm/scratch/flexicodec/speech_ft_checkpoint/finetuned_mls_checkpoints_from_multidomain/ft-just-sem-quant_{}_including-sem-rvq.th/checkpoint.th"
#     init_model_ckpt = "/home/poonehm/scratch/flexicodec/new_multidomain_checkpoint/finetune/encodec_checkpoints/encodec_multi_domain.th"
    
#     # Format the paths
#     ckpt_paths = [ckpt_fp.format(name) for name in task_names]

#     logger.info("Initializing Probing Model...")
#     # Assuming 8 classes for your downstream task, and FlexiCodec outputs dim 1024
#     model = DownstreamProbe(
#         init_model_ckpt=init_model_ckpt,
#         ckpt_paths=ckpt_paths,
#         merge_prefixes=MERGE_PREFIXES,
#         num_classes=8,
#         flexi_dim=128,
#         llm_dim=4096
#     ).cuda()
# # ... after initializing model = DownstreamProbe(...) ...

#     logger.info("\n--- VERIFICATION STEP 3: Mathematical Identity Test ---")
    
#     # 1. Artificially force the Softmax to output [1.0, 0.0]
#     # We do this by making the first weight massive.
#     with torch.no_grad():
#         model.merging_adapter.task_weights[0] = 100.0 
#         model.merging_adapter.task_weights[1] = -100.0
    
#     # 2. Ask the adapter to generate the mixed weights based on these fake probabilities
#     test_dynamic_params = model.merging_adapter.get_dynamic_state_dict()
    
#     # 3. Pick a specific target layer to test (e.g., the first one we matched)
#     test_key = model.merging_adapter.target_keys[0]
#     buffer_name = model.merging_adapter.finetuned_buffers[test_key]
    
#     # 4. Get the raw Task 0 weight directly from the stacked buffer
#     raw_task0_weight = getattr(model.merging_adapter, buffer_name)[0] # Index 0 is wavlm
    
#     # 5. Get the dynamically mixed weight
#     mixed_weight = test_dynamic_params[test_key]
    
#     # 6. Compare them! They should be 100% identical.
#     is_identical = torch.allclose(raw_task0_weight, mixed_weight, atol=1e-6)
    
#     if is_identical:
#         logger.info(f"SUCCESS: When Task 0 probability is 1.0, '{test_key}' perfectly matches the Task 0 checkpoint!")
#     else:
#         logger.error(f"FAILURE: The dynamically mixed weights do not match the target checkpoint!")
        
#     # Reset the task weights back to equal [1.0, 1.0] for actual training!
#     with torch.no_grad():
#         model.merging_adapter.task_weights.fill_(1.0)
        
#     logger.info("Identity test complete. Weights reset. Ready for training.\n")
#     optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

#     logger.info("Creating dummy audio tensor...")
#     # Dummy audio: Batch=2, Channels=1, Samples=32000 (e.g., ~0.7 seconds at 44.1kHz)
#     dummy_audio = torch.randn(2, 1, 32000).cuda()
#     dummy_labels = torch.tensor([3, 7]).cuda()

#     logger.info("Running Forward Pass...")
#     logits = model(dummy_audio)
    
#     logger.info("Running Backward Pass...")
#     loss = nn.CrossEntropyLoss()(logits, dummy_labels)
#     loss.backward()
#     optimizer.step()

#     logger.info("Test Complete! Checking gradients and mixing weights...")
#     mix_probs = torch.softmax(model.merging_adapter.task_weights, dim=0)
    
#     print(f"\nSUCCESS! Output logits shape: {logits.shape}")
#     print(f"Current Task Mixing Probabilities: {mix_probs.tolist()}")
#     print(f"Gradient flowing to task weights: {model.merging_adapter.task_weights.grad is not None}")