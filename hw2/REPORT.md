# HW2 — оптимизация обучения и ring collectives

## 1. Экспериментальный сетап

- GPU: NVIDIA A100 80 GB PCIe
- Модель: Qwen3, ~961M параметров
- Токенизатор: `ai-forever/rugpt3small_based_on_gpt2`
- Датасет: `wikimedia/wikipedia`, конфигурация `20231101.ru`
- Максимальная длина последовательности: 512
- Precision: BF16
- Optimizer: AdamW fused
- Learning rate: `3e-4`
- Scheduler: constant with warmup
- Warmup: 200 шагов
- Seed: 42
- Batch size: 32
- Gradient accumulation: 2
- Все остальные параметры и train/validation split сохранены как в HW1.
- Для коротких экспериментов использовалось 90 секунд, для baseline и финальной конфигурации — 300 секунд.

Главная метрика — useful tokens/sec.

## 2. Результаты экспериментов

| Конфигурация | Время, с | Tokens/sec | Peak GPU, GiB | Val loss | Optimizer steps |
|---|---:|---:|---:|---:|---:|
| Baseline | 300.8 | 6,769 | 41.24 | 6.685 | 354 |
| FlashAttention2 | 90.2 | 7,700 | 39.30 | 8.099 | 119 |
| Gradient checkpointing | 90.4 | 6,022 | 17.39 | 8.236 | 93 |
| Packing | 90.2 | 7,522 | 46.81 | 10.791 | 21 |
| Padding-free | 90.3 | 14,711 | 15.63 | 7.358 | 231 |
| Padding-free + FlashAttention2 | 90.1 | **19,381** | 15.11 | 6.884 | 304 |
| Padding-free + GC + FlashAttention2 | 90.0 | 15,715 | 9.63 | 7.259 | 246 |
| **Финальная: Padding-free + FlashAttention2** | **300.3** | **18,225** | **15.18** | **5.150** | **953** |

`torch.compile` и Liger kernels не использовались в итоговой конфигурации: их запуск завершался ошибкой Triton/CUDA `device kernel image is invalid`.

## 3. Итоговое ускорение

Для baseline:

`tokens_per_second = 6,769.30`

Для финальной конфигурации:

`tokens_per_second = 18,224.67`

Ускорение:

`S = 18,224.67 / 6,769.30 ≈ 2.69×`

Таким образом, финальная конфигурация увеличила скорость обучения примерно в **2.69 раза**.

## 4. Наблюдения

1. **Padding-free дал основной прирост производительности.**  
   Скорость выросла с 6,769 до 14,711 tokens/sec, то есть более чем в 2 раза. Одновременно пиковая выделенная память снизилась с 41.24 до 15.63 GiB.

2. **FlashAttention2 сам по себе дал небольшой прирост.**  
   Скорость увеличилась с 6,769 до 7,700 tokens/sec, а память немного снизилась.

3. **Gradient checkpointing уменьшил потребление памяти, но замедлил обучение.**  
   Память снизилась до 17.39 GiB, однако скорость упала до 6,022 tokens/sec. В комбинации с padding-free и FlashAttention2 скорость составила 15,715 tokens/sec — ниже, чем без checkpointing, но память снизилась до 9.63 GiB.

4. **Комбинация padding-free + FlashAttention2 оказалась лучшей.**  
   На коротком запуске она дала 19,381 tokens/sec. В финальном 300-секундном запуске получено 18,225 tokens/sec, что соответствует ускорению 2.69× относительно baseline.

5. **Packing не дал преимущества в данном эксперименте.**  
   Скорость составила 7,522 tokens/sec, а пиковая память выросла до 46.81 GiB. Результат также основан только на 11 измеренных шагах и поэтому менее стабилен.

## 5. Ring collectives

В `collectives.py` реализованы:

- `ring_reduce_scatter`
- `ring_all_gather`
- `ring_all_reduce`

Корректность проверена сравнением с PyTorch `torch.distributed`:

- 2 процесса — все три операции прошли проверку;
- 4 процесса — все три операции прошли проверку.

Проверялось также отсутствие изменения исходных входных тензоров.

## 6. Итог

Лучшая конфигурация:

- FlashAttention2
- Padding-free
- Gradient checkpointing: выключен
- Batch size: 32
- Gradient accumulation: 2
- BF16
- AdamW fused

Итоговая скорость: **18,224.67 useful tokens/sec**.  
Ускорение относительно baseline: **2.69×**.  
Пиковая выделенная GPU-память: **15.18 GiB**.  
Validation loss: **5.1495**.