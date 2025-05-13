import torch
import megatron
import megatron
from megatron import get_args, initialize_megatron
from megatron.model import ModelType
from megatron.models import TransformerModel
from megatron.utils import get_ltor_masks_and_position_ids
from transformers import QwenTokenizer

# ==============================
# 单卡训练参数配置（硬编码）
# ==============================
def main():
    # 禁用分布式并行（单卡专属配置）
    args = megatron.argparser.get_args()
    args.tensor_model_parallel_size = 1
    args.pipeline_model_parallel_size = 1
    args.data_parallel_size = 1
    args.micro_batch_size = 8           # 单卡微批量大小（可根据显存调整）
    args.global_batch_size = 8
    args.num_layers = 24                 # Qwen-0.5B层数
    args.hidden_size = 2048              # 隐藏层维度
    args.num_attention_heads = 32        # 注意力头数
    args.seq_length = 2048               # 序列长度
    args.vocab_size = 32000              # Qwen词表大小
    args.max_steps = 100                  # 简化训练步数（实际训练需增大）
    args.lr = 1e-4                       # 学习率
    args.fp16 = True                      # 启用混合精度
    args.checkpoint_activations = True   # 梯度检查点（节省显存）
    
    # 初始化Megatron（单卡模式）
    initialize_megatron(args=args)
    
    # ==============================
    # 构建模型
    # ==============================
    model = TransformerModel(
        num_layers=args.num_layers,
        hidden_size=args.hidden_size,
        num_attention_heads=args.num_attention_heads,
        vocab_size=args.vocab_size,
        seq_length=args.seq_length,
        model_type=ModelType.encoder_decoder  # Decoder-only模型兼容模式
    ).cuda()
    
    # ==============================
    # 初始化分词器和模拟数据
    # ==============================
    tokenizer = QwenTokenizer.from_pretrained("qwen-2.5-0.5b")
    
    # 模拟训练数据（生成随机输入）
    def generate_random_batch(batch_size, seq_length):
        input_ids = torch.randint(
            low=0, 
            high=args.vocab_size, 
            size=(batch_size, seq_length), 
            dtype=torch.long
        ).cuda()
        attention_mask = torch.ones_like(input_ids).cuda()
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask
        }
    
    # ==============================
    # 优化器和混合精度设置
    # ==============================
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=args.lr, 
        weight_decay=0.01
    )
    
    scaler = torch.cuda.amp.GradScaler(enabled=args.fp16)
    
    # ==============================
    # 训练循环
    # ==============================
    for step in range(args.max_steps):
        # 生成模拟批次
        batch = generate_random_batch(
            batch_size=args.micro_batch_size,
            seq_length=args.seq_length
        )
        
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        
        # 获取位置ID和掩码（Megatron需要的格式）
        _, position_ids = get_ltor_masks_and_position_ids(
            input_ids, 
            args.pad_token_id, 
            args.eod_token_id, 
            args.seq_length
        )
        
        # 混合精度前向传播
        with torch.cuda.amp.autocast(enabled=args.fp16):
            logits = model(
                input_ids, 
                position_ids=position_ids, 
                attention_mask=attention_mask
            )
            
            # 计算损失（假设语言模型训练）
            labels = input_ids[:, 1:].contiguous()
            logits = logits[:, :-1, :]
            loss = torch.nn.functional.cross_entropy(
                logits.view(-1, args.vocab_size), 
                labels.view(-1)
            )
        
        # 反向传播
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        
        # 打印训练信息
        if step % 10 == 0:
            print(f"Step {step}, Loss: {loss.item():.4f}")
    
    # 保存单卡模型（无需分片）
    torch.save(model.state_dict(), "qwen_0.5b_single_card.pth")
    print("训练完成，模型已保存")

if __name__ == "__main__":
    # 确保单卡环境
    torch.cuda.set_device(0)
    main()