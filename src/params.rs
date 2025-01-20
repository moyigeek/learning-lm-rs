use crate::config::LlamaConfigJson;
use crate::tensor::{self, Tensor};
use safetensors::SafeTensors;
pub struct LLamaParams<T> {
    // token_id to embedding lookup table
    pub embedding_table: Tensor<T>, // (vocab_size, dim)
    // decoder layer
    pub rms_att_w: Vec<Tensor<T>>, // (hidden_size, ) x layers
    pub wq: Vec<Tensor<T>>,        // (n_heads * head_size, hidden_size) x layers
    pub wk: Vec<Tensor<T>>,        // (n_kv_heads * head_size, hidden_size) x layers
    pub wv: Vec<Tensor<T>>,        // (n_kv_heads * head_size, hidden_size) x layers
    pub wo: Vec<Tensor<T>>,        // (hidden_size, n_heads * head_size) x layers
    // ffn layer
    pub rms_ffn_w: Vec<Tensor<T>>, // (hidden_size, ) x layers
    pub w_up: Vec<Tensor<T>>,      // (intermediate_size, hidden_size) x layers
    pub w_gate: Vec<Tensor<T>>,    // (intermediate_size, hidden_size) x layers
    pub w_down: Vec<Tensor<T>>,    // (hidden_size, intermediate_size) x layers
    // output
    pub rms_out_w: Tensor<T>, // (hidden_size, )
    pub lm_head: Tensor<T>,   // (vocab_size, dim)
}

impl LLamaParams<f32> {
    pub fn from_safetensors(safetensor: &SafeTensors, config: &LlamaConfigJson) -> Self {
        // todo!("实现从safetensors文件的模型参数加载");
        // let get_tensor: impl Fn(&str) -> Tensor<f32> = |name: &str| {
        // ...    
        // };
        let get_tensor=|name: &str|->Tensor<f32>{
            let tensor=safetensor.tensor(name).expect(&format!("{} not found",name));
            let data_bytes = tensor.data();
            if data_bytes.len() % 4 != 0 {
                panic!("Data length is not a multiple of 4 bytes for f32 conversion");
            }
            let data = data_bytes
                .chunks(4)
                .map(|chunk| f32::from_le_bytes(chunk.try_into().expect("Chunk size incorrect")))
                .collect::<Vec<f32>>();
            let shape = tensor.shape().to_vec();
            Tensor::new(data, &shape)
            
        };
        let n_layers = config.num_hidden_layers;
        let names=safetensor.names();
        for name in names{
            println!("{}",name);
        };
        
        // LLamaParams {
        //     embedding_table: get_tensor(...),
        //     ...
        // }
        LLamaParams {
            // Update the tensor name based on actual safetensors names
            embedding_table: get_tensor("lm_head.weight"),
            rms_att_w: (0..n_layers)
                .map(|i| get_tensor(&format!("model.layers.{i}.input_layernorm.weight")))
                .collect(),
            wq: (0..n_layers)
                .map(|i| get_tensor(&format!("model.layers.{i}.self_attn.q_proj.weight")))
                .collect(),
            wk: (0..n_layers)
                .map(|i| get_tensor(&format!("model.layers.{i}.self_attn.k_proj.weight")))
                .collect(),
            wv: (0..n_layers)
                .map(|i| get_tensor(&format!("model.layers.{i}.self_attn.v_proj.weight")))
                .collect(),
            wo: (0..n_layers)
                .map(|i| get_tensor(&format!("model.layers.{i}.self_attn.o_proj.weight")))
                .collect(),
            rms_ffn_w: (0..n_layers)
                .map(|i| get_tensor(&format!("model.layers.{i}.post_attention_layernorm.weight")))
                .collect(),
            w_up: (0..n_layers)
                .map(|i| get_tensor(&format!("model.layers.{i}.mlp.up_proj.weight")))
                .collect(),
            w_gate: (0..n_layers)
                .map(|i| get_tensor(&format!("model.layers.{i}.mlp.gate_proj.weight")))
                .collect(),
            w_down: (0..n_layers)
                .map(|i| get_tensor(&format!("model.layers.{i}.mlp.down_proj.weight")))
                .collect(),
            rms_out_w: get_tensor("model.norm.weight"),
            lm_head: get_tensor("lm_head.weight"),
        }
    }
}
