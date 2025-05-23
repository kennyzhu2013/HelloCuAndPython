#pragma once

#include "codec.h"
#include "cuda_runtime_api.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <memory>
#include <vector>
#include <map>

#define DEBUG_MODEL 0

constexpr int KV_SINKS = 2;

enum class ActivationType {
    GELU,
    SILU,
};

enum class LayerNormType {
    RMSNorm,
};

// TODO:华为的device待增加适配：CANN
enum class Device {
    CPU,
    CUDA,
};

enum class InferenceMode {
    HYDRATE_KV_CACHE, // only hydrate the KV cache and don't compute output logits, 仅填充 KV 缓存，不计算输出的 logits
    OUTPUT_LOGITS // set InferenceState logits to logits for the next token
};

extern "C" void* upload_cuda(void* host, size_t size);
extern "C" void* download_cuda(void* device, size_t size, std::string debug);
extern "C" void register_cuda_host(void* host, size_t size);
extern "C" void free_cuda(void* device);
extern "C" void unregister_cuda_host(void* host);
extern "C" void set_cuda_device(int device);
extern "C" void init_cuda_stream(cudaStream_t * stream);

struct Config {
    int dim;                  // Transformer 输入 & 输出维度
    int hidden_dim;           // 前馈网络隐藏层的维度
    int head_dim;             // 每个注意力头的维度，通常 `dim / n_heads`
    int n_layers;             // number of layers 层数
    int n_heads;              // number of attention query heads 注意力查询头数
    int n_kv_heads;           // 键值头数number of key and value heads; can be < n_heads (1 is MultiQueryAttention, >1 is GroupedQueryAttention)
    int vocab_size;           // 词汇表大小vocabulary size
    int max_seq_len;          // 最大序列长度
    float rope_theta;         // RoPE 位置编码参数RoPE theta
    int rotary_dim;           // 旋转位置编码维度dimension of rotary position encoding (elements after that don't get rotated)
    float norm_eps;           // 层归一化的 epsilon 值epsilon for layer normalization
    ActivationType act;       // 激活函数类型activation function
    LayerNormType norm_type;  // 归一化类型norm type
    float qkv_clip;           // clip qkv values to [-clip, clip]
    // mixture of experts，专家（激活）模型数量
    int n_experts;
    int n_experts_active;

    // Data type of the weights according to config, used
    // to safety check tensor dtype at initialization time.权重数据类型
    DType weight_dtype;

    // context: If nonzero `context` is supplied, max sequence length is limited to `context`.
    void from_yalm(YALMData& yalm, int context = 0);
    size_t active_bytes(size_t pos) const;
};

// 这段代码定义了一个名为 CudaGraph 的结构体，主要用于管理 CUDA 图形（CUDA Graphs）。
// CUDA 图形是一种用于优化 CUDA 应用程序性能的机制，通过将多个 CUDA 操作组合成一个图形并在其上执行，可以减少 CPU 和 GPU 之间的调度开销。
struct CudaGraph {
    cudaGraph_t graph;    // 这是一个 CUDA 图形对象，表示整个图形的结构。它用于存储和管理图形中的所有节点和操作
    cudaGraphExec_t instance;  // 这是一个 CUDA 图形执行实例. 表示图形的具体执行状态,通过这个实例，可以在 CUDA 流中执行之前定义好的图形。
    bool is_created = false;  // 用于指示图形是否已成功创建
    std::unordered_map<std::string, cudaGraphNode_t> nodes;  // 这是一个哈希表，用于存储图形中的节点。每个节点都与一个字符串键（key）相关联，cudaGraphNode_t 是 CUDA 中表示图形节点的类型

    // 这个函数接受一个函数对象（func）和一个 CUDA 流（s）作为参数。它的作用是将给定的 CUDA 操作(作为一个节点)封装到图形中，以便后续执行。
    void wrap(std::function<void()> func, cudaStream_t s);
    void launch(cudaStream_t s);  // 启动执行
    // 用于向图形中添加或更新一个内核节点。它接受一个字符串 key 作为标识符，params 用于传递内核参数，stream 是指定的 CUDA 流。
    // 该函数的实现可能会检查是否已经存在具有相同键的节点，如果存在则更新该节点，否则添加一个新节点。
    void add_or_update_kernel_node(std::string key, cudaKernelNodeParams params, cudaStream_t stream);
};

// Buffer for all state used during a forward pass.
// Members are reused across subsequent blocks and passes.
// This lets us avoid allocations during inference.
// 推理状态：它存储了神经网络在推理过程中的激活状态和 logits（最终输出），让我们避免多次分配内存，提升效率。
struct InferenceState {
    InferenceState(const std::shared_ptr<Config> config);
    ~InferenceState();

    // current activations
    float* x() const { return _x; }
    float* xb() const { return _xb; }
    float* xb(int head) const { return _xb + _config->head_dim * head; }  // 返回特定头的激活值
    // TODO: do we need xb2?
    float* xb2() const { return _xb2; }   // 返回第二个残差分支的激活值
    float* xb2(int head) const { return _xb2 + _config->head_dim * head; }  // 返回特定头的第二个残差分支的激活值

    float* hb() const { return _hb; }  // 返回隐藏层的缓冲区，用于前馈网络
    float* hb2() const { return _hb2; }  // 返回第二个隐藏层的缓冲区
    float* q() const { return _q; }  // 返回最新时间戳的查询向量
    float* q(int head) const { return _q + _config->head_dim * head; }
    float* k() const { return _k; }  // 返回最新时间戳的键向量
    float* v() const { return _v; }
    float* att() const { return _att; }  // 返回注意力分数的缓冲区
    float* att(int head) const { return _att + _config->max_seq_len * head; }  // 返回特定头的注意力分数缓冲区
    // mixture of experts
    float* moe_weights() const { return _moe_weights; }
    float* active_experts_weights() const { return _active_experts_weights; }
    int* active_experts() const { return _active_experts; }
    // LM head, logits输出
    float* logits() const { return _logits; }

    void cuda();
    Device device() const { return _device; }
    cudaStream_t stream() const { return _stream; }
    InferenceMode mode() const { return _mode; }
    void set_mode(InferenceMode mode) { _mode = mode; }

    // 根据当前的推理模式返回相应的 CUDA 图（_hydrate_graph 或 _output_graph）。
    CudaGraph& graph() {
        return _mode == InferenceMode::HYDRATE_KV_CACHE ? _hydrate_graph : _output_graph;
    }

private:
    std::shared_ptr<Config> _config;
    Device _device = Device::CPU;
    cudaStream_t _stream;
    InferenceMode _mode = InferenceMode::OUTPUT_LOGITS;

    // 针对不同InferenceMode的两种CudaGraph
    CudaGraph _hydrate_graph;
    CudaGraph _output_graph;

    // 以下指针用于存储模型在推理过程中的各种激活状态和输出。它们指向在 GPU 或 CPU 上分配的内存，根据模型的需求进行动态管理。
    // current activations
    float* _x = nullptr;         // (dim,) - latest activation
    float* _xb = nullptr;        // (dim,) - activation inside a residual branch
    // TODO: do we need xb2?
    float* _xb2 = nullptr;       // (dim,) - activation inside a residual branch (second slot)
    float* _hb = nullptr;        // (hidden_dim,) - buffer for hidden dimension in feedforward network
    float* _hb2 = nullptr;       // (hidden_dim,) - buffer for hidden dimension in feedforward network (second slot)
    float* _q = nullptr;         // (n_heads * head_dim,) - query vectors for latest timestamp
    float* _k = nullptr;         // (n_kv_heads * head_dim,) - key vectors for latest timestamp
    float* _v = nullptr;         // (n_kv_heads * head_dim,) - value vectors for latest timestamp
    float* _att = nullptr;       // (n_heads, seq_len) - buffer for attention scores
    // mixture of experts
    float* _moe_weights = nullptr; // (n_experts,) - buffer for expert weights, decided by router
    float* _active_experts_weights = nullptr; // (n_active_experts,) - buffer for weights of top K experts (active experts)
    int* _active_experts = nullptr; // (n_active_experts,) - buffer for indices of top K experts (active experts)

    // LM head
    // NOTE: this always lives on the host (CPU), but must be registered 
    // with CUDA to be used on the device.
    float* _logits = nullptr;    // (vocab_size,) - final output logits
};

/* Transformer Block：Transformer 模型块 ：主要用于实现 Transformer 模型中的一个计算块。每个块包含自注意力机制、前馈网络以及层归一化等功能。*/
struct Block {
    Block(
        int layer_i,
        const std::shared_ptr<Config> config,
        const Tensor* rms_att_weight,
        const Tensor* rms_ffn_weight,
        const Tensor* wq,
        const Tensor* wk,
        const Tensor* wv,
        const Tensor* wo,
        const Tensor* w1,
        const Tensor* w2,
        const Tensor* w3,
        const Tensor* moegate
    );
    ~Block();

    float* rms_att_weight() const { return _rms_att_weight; }  // 返回自注意力的 RMS 权重
    float* rms_ffn_weight() const { return _rms_ffn_weight; }  // 返回前馈网络的 RMS 权重
    template <typename T>
    T* wq() const { return static_cast<T*>(_wq); }
    template <typename T>
    T* wk() const { return static_cast<T*>(_wk); }
    template <typename T>
    T* wv() const { return static_cast<T*>(_wv); }
    template <typename T>
    T* wo() const { return static_cast<T*>(_wo); }
    template <typename T>
    T* w1() const { return static_cast<T*>(_w1); }
    template <typename T>
    T* w2() const { return static_cast<T*>(_w2); }
    template <typename T>
    T* w3() const { return static_cast<T*>(_w3); }
    template <typename T>
    T* moegate() const { return static_cast<T*>(_moegate); }
    f16_t* key_cache() const { return _key_cache; }
    f16_t* value_cache() const { return _value_cache; }

    // Compute forward pass for this block and update the inference state accordingly.
    // PRECONDITIONS: 
    // - `s.x()` contains the input to the block. Output will also go here.
    // - Block KV cache is hydrated.
    // 这是 Transformer 的核心计算单元，每个块包含：
    // 自注意力机制（Multi - Head Attention）
    // 前馈网络（Feedforward Network）
    // 层归一化（Layer Normalization） 每个 Block 处理一个 Transformer 层的数据，并使用 InferenceState 记录计算结果。
    void block(
        InferenceState& s,  // inference state
        int pos,            // index of the current token in the sequence
        int kv_sink,        // number of sink tokens currently in the KV cache
        int kv_pos,         // index of the current token in the kv cache, must be in [0..kv_len) since kv cache is a ring buffer
        int kv_len          // number of tokens in the kv cache that we will attend over
    ) const;

    void cuda();

private:
    template <typename T>
    void _block_cpu(
        InferenceState& s,  // inference state
        int pos,            // index of the current token in the sequence
        int kv_sink,        // number of sink tokens currently in the KV cache
        int kv_pos,         // index of the current token in the kv cache, must be in [0..kv_len) since kv cache is a ring buffer
        int kv_len          // number of tokens in the kv cache that we will attend over
    ) const;
    template <typename T>
    void _block_cuda(
        InferenceState& s,  // inference state
        int pos,            // index of the current token in the sequence
        int kv_sink,        // number of sink tokens currently in the KV cache
        int kv_pos,         // index of the current token in the kv cache, must be in [0..kv_len) since kv cache is a ring buffer
        int kv_len          // number of tokens in the kv cache that we will attend over
    ) const;

    int _layer_i = 0;  // 当前层的索引

    std::shared_ptr<Config> _config;
    Device _device = Device::CPU;

    // weights for norms
    float* _rms_att_weight = nullptr; // (dim) rmsnorm weights
    float* _rms_ffn_weight = nullptr; // (dim)

    // weights for self-attention matmuls
    void* _wq = nullptr; // (n_heads * head_dim, dim)
    void* _wk = nullptr; // (n_kv_heads * head_dim, dim)
    void* _wv = nullptr; // (n_kv_heads * head_dim, dim)
    void* _wo = nullptr; // (dim, n_heads * head_dim)

    // weights for ffn
    void* _w1 = nullptr; // (n_experts?, hidden_dim, dim)
    void* _w2 = nullptr; // (n_experts?, dim, hidden_dim)
    void* _w3 = nullptr; // (n_experts?, hidden_dim, dim) - GLU weights
    // weights for mixture of experts router if present
    void* _moegate = nullptr; // (n_experts?, dim)

    // kv cache
    f16_t* _key_cache = nullptr;   // (seq_len, n_kv_heads * head_dim)
    f16_t* _value_cache = nullptr; // (seq_len, n_kv_heads * head_dim)
};

struct Model {
    std::shared_ptr<Config> config;

    std::vector<std::shared_ptr<Block>> blocks;

    // token embedding table
    void* token_embedding_table = nullptr; // (vocab_size, dim)
    // final norm
    float* rms_final_weight = nullptr; // (dim,)
    // classifier weights for the logits, on the last layer
    void* wcls = nullptr; // (vocab_size, dim)

    Model(YALMData& yalm, int context = 0);

    void forward(InferenceState& s, int token, int pos, InferenceMode mode = InferenceMode::OUTPUT_LOGITS);
    void cuda();

private:
    void _forward_cpu(InferenceState& s, int token, int pos, InferenceMode mode);
    void _forward_cuda(InferenceState& s, int token, int pos, InferenceMode mode);
    void _forward_cuda_build_graph(InferenceState& s, int token, int pos, InferenceMode mode);
    void _copy_embedding(InferenceState& s, int token);

    Device _device = Device::CPU;
};

#if DEBUG_MODEL
struct DebugTensor {
    enum struct DataType {
        F32,
        F16,
    };

    DebugTensor() = default;
    DebugTensor(const std::vector<float>& data);
    DebugTensor(const std::vector<f16_t>& data);
    DebugTensor& operator=(const DebugTensor& other) = default;
    float max_err(const DebugTensor& other) const;

    std::vector<float> data_f32;
    std::vector<f16_t> data_f16;
    DataType data_type;
};
std::map<std::string, DebugTensor>& debug_map_cpu();
std::map<std::string, DebugTensor>& debug_map_cuda();
#endif

////////////////////////////////////////
// Exposed for tests
////////////////////////////////////////
void attn(
    float* xout,    // (dim,) - output vector
    float* atth,    // (kv_len,) - scratch space to hold attention scores of the sequence
    float* qh,      // (head_dim,) - query vector for this head
    f16_t* kh,      // (kv_len, n_kv_heads, head_dim) - buffer containing key vectors of the sequence for all KV heads
    f16_t* vh,      // (kv_len, n_kv_heads, head_dim) - buffer containing value vectors of the sequence for all KV heads
    int head_dim,   // size of the "key-space"
    int n_kv_heads, // number of kv heads, can be < n_heads (1 is MultiQueryAttention, >1 is GroupedQueryAttention)
    int kv_len      // number of tokens of the sequence we will attend over
);

void mha_cpu(
    float* xout,  // (n_heads, head_dim)
    float* att,   // (n_heads, max_seq_len)
    f16_t* kb,    // (max_seq_len, n_kv_heads, head_dim)
    f16_t* vb,    // (max_seq_len, n_kv_heads, head_dim)
    float* q,     // (n_heads, head_dim)
    int head_dim, int kv_len, int max_seq_len, int n_heads, int n_kv_heads
);
void mha_cuda(
    float* xout,  // (n_heads, head_dim)
    float* att,   // (n_heads, max_seq_len)
    f16_t* kb,    // (max_seq_len, n_kv_heads, head_dim)
    f16_t* vb,    // (max_seq_len, n_kv_heads, head_dim)
    float* q,     // (n_heads, head_dim)
    int head_dim, int kv_len, int max_seq_len, int n_heads, int n_kv_heads
);

void matmul_cpu(float* xout, float* x, float* w, int n, int d);
void matmul_cpu(float* xout, float* x, f16_t* w, int n, int d);
template <typename T>
void matmul_cuda(float* xout, float* x, T* w, int n, int d);

void ffn_cpu(
    float* xout, float* x,
    float* w1, float* w2, float* w3,
    int hidden_dim, int dim,
    ActivationType act
);
template <typename T>
void ffn_cuda(
    float* xout, float* x,
    T* w1, T* w2, T* w3,
    int hidden_dim, int dim,
    ActivationType act
);
////////////////////////////////////////
