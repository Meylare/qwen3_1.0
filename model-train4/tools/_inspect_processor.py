import sys
from pathlib import Path

from transformers import AutoProcessor

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from train_gspo_qwen35_smoke import resolve_model_path, ensure_chat_template, DEFAULT_MODEL_DIR
model=resolve_model_path(str(DEFAULT_MODEL_DIR))
processor=AutoProcessor.from_pretrained(model, trust_remote_code=True)
ensure_chat_template(processor, model)
vp=getattr(processor,'video_processor', None)
ip=getattr(processor,'image_processor', None)
print('processor', processor.__class__.__name__)
print('video_processor', None if vp is None else vp.__class__.__name__)
print('image_processor', None if ip is None else ip.__class__.__name__)
obj=vp or ip
for name in ['size','min_pixels','max_pixels','patch_size','temporal_patch_size','merge_size','do_resize','do_rescale','do_normalize','resample']:
    print(name, getattr(obj, name, None))
print('attrs sample', sorted([k for k in vars(obj).keys() if not k.startswith('_')])[:80])
