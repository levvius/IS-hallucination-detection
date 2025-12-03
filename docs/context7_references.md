# Context7 Documentation References

## Overview
This document contains key findings from Context7 documentation for FastAPI, sentence-transformers, transformers, and FAISS libraries to guide code improvements.

## 1. FastAPI - Exception Handling

### Key Patterns

**Custom Exception Handlers:**
```python
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

class AppBaseException(Exception):
    def __init__(self, message: str, details: dict = None):
        self.message = message
        self.details = details or {}

@app.exception_handler(AppBaseException)
async def app_exception_handler(request: Request, exc: AppBaseException):
    return JSONResponse(
        status_code=500,
        content={"detail": exc.message, **exc.details}
    )
```

**Best Practices:**
- Use `@app.exception_handler()` decorator for global exception handling
- Return `JSONResponse` with appropriate status codes (400, 404, 500, 503)
- Include custom headers for advanced scenarios: `headers={"X-Error": "..."}`
- Reuse default handlers: `from fastapi.exception_handlers import http_exception_handler`
- Override `HTTPException` and `RequestValidationError` handlers for custom responses

**Status Code Guidelines:**
- 400: Bad Request / Input Validation
- 404: Not Found
- 418: Custom status (teapot example)
- 500: Internal Server Error
- 503: Service Unavailable (models not loaded)

---

## 2. Sentence Transformers - Model Lifecycle

### Key Patterns

**Model Loading:**
```python
from sentence_transformers import SentenceTransformer

# Default PyTorch backend (auto device selection)
model = SentenceTransformer("all-MiniLM-L6-v2")

# Explicit backend selection (ONNX for optimization)
model = SentenceTransformer(
    "all-MiniLM-L6-v2",
    backend="onnx",
    model_kwargs={"file_name": "onnx/model_O3.onnx"}
)
```

**Encoding Best Practices:**
```python
# Basic encoding
sentences = ["Example sentence 1", "Example sentence 2"]
embeddings = model.encode(sentences)

# Batch processing (implicit)
# SentenceTransformer handles batching internally
```

**Optimization Techniques:**
- Use ONNX backend for production: `backend="onnx"`
- Quantization for reduced memory: `model_kwargs={"file_name": "onnx/model_qint8.onnx"}`
- OpenVINO for Intel CPUs: `backend="openvino"`
- Models automatically select best device (CUDA > MPS > CPU)

**Important:**
- Models are stateful - load once, reuse across requests
- Use singleton pattern for model management (already implemented)
- No explicit device management needed with PyTorch backend

---

## 3. Transformers (HuggingFace) - Pipeline & Device Management

### Key Patterns

**Pipeline Creation with Device:**
```python
from transformers import pipeline
import torch

# Explicit GPU device
pipe = pipeline(
    task="text-generation",
    model="meta-llama/Meta-Llama-3-8B",
    device=0  # GPU 0
)

# Auto device mapping (recommended)
pipe = pipeline(
    task="text-generation",
    model="meta-llama/Meta-Llama-3-8B",
    device_map="auto",
    dtype=torch.bfloat16
)

# Apple Silicon (MPS)
pipe = pipeline(
    task="text-generation",
    model="google/gemma-2-2b",
    device="mps"
)

# CPU explicitly
pipe = pipeline(
    task="text-generation",
    model="google/gemma-2-2b",
    device=-1
)
```

**Error Handling:**
- Use `device_map="auto"` for automatic device selection
- Catch model loading errors early at startup
- Handle `torch.cuda.OutOfMemoryError` for GPU issues
- Use `dtype` optimization: `torch.bfloat16` or `torch.float16`

**Best Practices:**
- Load pipeline once at startup (singleton pattern)
- Use `device_map="auto"` for production
- Install `accelerate` for device management: `pip install -U accelerate`
- Batch processing: pass list of inputs to pipeline

**Batch Processing:**
```python
from accelerate import Accelerator

device = Accelerator().device
pipe = pipeline(task="text-generation", model="model-name", device=device)

# Process multiple inputs
outputs = pipe(["input 1", "input 2", "input 3"])
```

---

## 4. FAISS - Index Optimization

### Key Patterns

**Index Types:**

**1. IndexFlatL2 (Exact Search, Baseline):**
```python
import faiss
import numpy as np

d = 64  # dimension
index = faiss.IndexFlatL2(d)
index.add(xb)  # add vectors
D, I = index.search(xq, k=4)  # search
```

**2. IndexIVFFlat (Inverted File, Fast):**
```python
nlist = 100  # number of cells
quantizer = faiss.IndexFlatL2(d)
index = faiss.IndexIVFFlat(quantizer, d, nlist)

# Training required
index.train(xb)
index.add(xb)

# Tune accuracy vs speed
index.nprobe = 10  # default=1, higher=more accurate
D, I = index.search(xq, k=4)
```

**3. IndexIVFPQ (Product Quantization, Memory Efficient):**
```python
nlist = 100
m = 8  # number of subquantizers
quantizer = faiss.IndexFlatL2(d)
index = faiss.IndexIVFPQ(quantizer, d, nlist, m, 8)  # 8 bits per subvector

index.train(xb)
index.add(xb)
index.nprobe = 10
D, I = index.search(xq, k=4)
```

**4. IndexPQFastScan (SIMD Optimized):**
```python
m = 8  # sub-vectors
n_bit = 4  # bits per sub-vector (4 or 6)
bbs = 32  # block size (multiple of 32)
index = faiss.IndexPQFastScan(d, m, n_bit, faiss.METRIC_L2, bbs)

index.train(xb)
index.add(xb)
D, I = index.search(xq, k=4)
```

**Dynamic Index Creation:**
```python
# Factory pattern for complex indexes
index = faiss.index_factory(d, "IVF100,Flat")
index = faiss.index_factory(d, "IVF100,PQ16")
index = faiss.index_factory(d, "PCA64,IVF100,PQ8")
index = faiss.index_factory(d, "HNSW32")
index = faiss.index_factory(d, "PQ16x4fs")  # FastScan
```

**Best Practices:**
- Use `IndexFlatL2` for <10k vectors (exact search)
- Use `IndexIVFFlat` for 10k-1M vectors (good speed/accuracy)
- Use `IndexIVFPQ` for >1M vectors (memory constrained)
- Use `IndexPQFastScan` for CPU with SIMD support
- **Always normalize vectors** for inner product: `faiss.normalize_L2(xb)`
- Train on representative data (>=30x nlist samples)
- Tune `nprobe` for accuracy: higher = slower but more accurate

**Current Implementation Issues:**
- Using `IndexFlatL2` (exact search) - OK for ~265 snippets
- Should add normalization for better results
- Consider `IndexIVFFlat` if KB grows >10k snippets

---

## Implementation Recommendations

### Priority 1: Exception Handling
- ✅ Implement custom exception hierarchy
- ✅ Use `@app.exception_handler()` for global handlers
- ✅ Return appropriate status codes (400, 503, 500)
- ✅ Include error details in response

### Priority 2: Model Management
- ✅ Current singleton pattern is correct
- ⚠️ Add explicit error handling in ModelManager
- ⚠️ Use `device_map="auto"` for NLI pipeline
- ✅ Load once at startup (already done)

### Priority 3: FAISS Optimization
- ⚠️ Add vector normalization before indexing
- ℹ️ Current IndexFlatL2 is fine for current scale
- ℹ️ Consider IndexIVFFlat if KB grows >10k

### Priority 4: Error Recovery
- Add try/except around all model operations
- Provide actionable error messages
- Log errors with context
- Return 503 for model-related errors

---

## References
- FastAPI: `/fastapi/fastapi` (Score: 87.8)
- Sentence Transformers: `/huggingface/sentence-transformers` (Score: 94.3)
- Transformers: `/huggingface/transformers` (Score: 72.3)
- FAISS: `/facebookresearch/faiss` (Score: 94.2)
