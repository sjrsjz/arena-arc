# arc-slice

一个针对类似 `Arc<[T]>` 的基于 chunk 的快速分配器，用于分配连续切片并通过句柄共享数据。适合高频短生命周期分配场景，避免频繁单独分配与释放。

## 特性

- 通过单个 chunk 分配多个切片
- 句柄可低成本克隆，数据零拷贝共享
- 支持可配置 chunk 容量与索引类型

## 快速开始

```rust
use arc_slice::Allocator;

let mut allocator: Allocator<u32, u32, 16> = Allocator::new();
let handle = allocator.alloc(4, |i| (i * 2) as u32);
assert_eq!(handle.get(), &[0, 2, 4, 6]);
```

## 安全性与行为说明

- **僵尸对象**：某个切片不再使用，但同一 chunk 仍被其他句柄持有时，该切片占用的内存不会被单独回收，直到整个 chunk 被释放。
- **额外内存占用**：chunk 以固定容量分配，可能存在未使用空间；当空间不足时会创建新 chunk，旧 chunk 的剩余空间会成为额外占用。
- **Drop 推迟**：元素的 `Drop` 只在最后一个持有该 chunk 的句柄释放后发生，不是逐元素即时释放。

## 基准测试

项目提供与标准库 `Arc<[T]>` 的分配对比基准。

```bash
cargo bench --bench alloc_bench
```

## Miri 检查

```bash
cargo +nightly miri test
```

## 许可证

MIT
